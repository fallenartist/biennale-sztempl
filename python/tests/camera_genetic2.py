#!/usr/bin/env python3
"""
IMX500 Genetic Tile Art - Portrait HD Display
Full-size crop/transformation for rotated HD TV in portrait orientation
Target size: 1064x1824 with proper margins

Key Features:
- Portrait HD display (1080x1920) with exact margins
- Genetic algorithm evolves B&W tiles to match pose-detected targets
- Multi-scale tiles: 152px (background), 76px (medium), 38px (face detail)
- Debug overlays: pose skeleton, posterized target, detail mask
- Real-time pose detection with crop region calculation
"""

import numpy as np
import pygame
import random
import math
import time
import cv2
import os
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
import sys
from typing import List, Tuple, Optional
import glob

# (The TileLibrary and GeneticMosaicGenerator classes are correct and remain unchanged)
class TileLibrary:
	def __init__(self, tiles_dir: str = "tiles"):
		self.tiles_dir, self.tile_variants, self.num_variants, self.base_tile_size = tiles_dir, [], 0, 20
		self.load_tiles()
	def _create_rotations(self, base_tiles: List[np.ndarray]) -> List[np.ndarray]:
		rotated_variants = []
		for tile in base_tiles:
			for angle in [0, 90, 180, 270]:
				center, M = (tile.shape[1]/2, tile.shape[0]/2), cv2.getRotationMatrix2D((tile.shape[1]/2, tile.shape[0]/2), angle, 1)
				rotated = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
				rotated_variants.append(rotated)
		return rotated_variants
	def load_tiles(self):
		tile_paths = glob.glob(os.path.join(self.tiles_dir, "*.png"))
		if not tile_paths: self.create_basic_tiles(); return
		base_tiles = []
		for path in sorted(tile_paths):
			try:
				img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
				if img is None: continue
				if img.ndim == 2: img = cv2.merge([img, img, img, np.full_like(img, 255)])
				elif img.shape[2] == 3: img = cv2.merge([*cv2.split(img), np.full_like(img[:,:,0], 255)])
				rgb, alpha = img[:,:,:3].astype(float), img[:,:,3:4].astype(float)/255.0
				composited = (rgb*alpha + 255*(1-alpha)).astype(np.uint8)
				gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
				_, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
				base_tiles.append(bw); print(f"Loaded {path}")
			except Exception as e: print(f"Error loading {path}: {e}")
		if not base_tiles: self.create_basic_tiles(); return
		self.base_tile_size = base_tiles[0].shape[0]
		base_tiles.extend([np.ones((self.base_tile_size, self.base_tile_size), dtype=np.uint8)*255, np.zeros((self.base_tile_size, self.base_tile_size), dtype=np.uint8)])
		self.tile_variants = self._create_rotations(base_tiles)
		self.num_variants = len(self.tile_variants); print(f"Created {self.num_variants} B&W tile variants")
	def create_basic_tiles(self):
		size, shapes = 20, []
		diamond = np.ones((size,size),dtype=np.uint8)*255; cv2.fillPoly(diamond,[np.array([[size//2,2],[size-2,size//2],[size//2,size-2],[2,size//2]])],0); shapes.append(diamond)
		triangle = np.ones((size,size),dtype=np.uint8)*255; cv2.fillPoly(triangle,[np.array([[size//2,2],[size-2,size-2],[2,size-2]])],0); shapes.append(triangle)
		circle = np.ones((size,size),dtype=np.uint8)*255; cv2.circle(circle,(size//2,size//2),size//2-2,0,-1); shapes.append(circle)
		cross = np.ones((size,size),dtype=np.uint8)*255; cv2.rectangle(cross,(size//2-2,2),(size//2+2,size-2),0,-1); cv2.rectangle(cross,(2,size//2-2),(size-2,size//2+2),0,-1); shapes.append(cross)
		shapes.extend([np.ones((size,size),dtype=np.uint8)*255, np.zeros((size,size),dtype=np.uint8)])
		self.tile_variants = self._create_rotations(shapes)
		self.num_variants = len(self.tile_variants); print(f"Created {self.num_variants} basic B&W tile variants")
	def render_variant(self, variant_idx:int, target_size:int) -> np.ndarray:
		tile = self.tile_variants[variant_idx % self.num_variants]
		return cv2.resize(tile, (target_size,target_size), interpolation=cv2.INTER_AREA) if target_size != self.base_tile_size else tile.copy()

class GeneticMosaicGenerator:
	def __init__(self, tile_library: TileLibrary, target_size: Tuple[int, int]):
		self.tile_library, self.target_size, self.scales, self.scale_thresholds = tile_library, target_size, [152,76,38], [80,180]
		self.regions, self.genome, self.mosaic, self.target_image, self.detail_mask = [],[],None,None,None
		self.best_score, self.max_iterations, self.current_iteration, self.WINDOW_SIZE_H_W = float('inf'), 30000, 0, None
	def create_pose_detail_mask(self, pose_keypoints:Optional[np.ndarray]) -> np.ndarray:
		mask = np.zeros(self.target_size[::-1], dtype=np.uint8)
		if pose_keypoints is None or len(pose_keypoints) == 0 or self.WINDOW_SIZE_H_W is None: return self.create_edge_detail_mask()
		try:
			best_person = pose_keypoints[0] if pose_keypoints.ndim == 3 else pose_keypoints
			if best_person.shape[0] < 17 or best_person.shape[1] != 3: return self.create_edge_detail_mask()
			h_scale, w_scale = self.target_size[1]/self.WINDOW_SIZE_H_W[0], self.target_size[0]/self.WINDOW_SIZE_H_W[1]
			for idx in [0,1,2,3,4]:
				if idx < len(best_person) and best_person[idx][2]>0.3: cv2.circle(mask,(int(best_person[idx][0]*w_scale),int(best_person[idx][1]*h_scale)),max(120,self.target_size[0]//12),255,-1)
			for idx in [5,6]:
				if idx < len(best_person) and best_person[idx][2]>0.3: cv2.circle(mask,(int(best_person[idx][0]*w_scale),int(best_person[idx][1]*h_scale)),max(100,self.target_size[0]//15),128,-1)
			valid_pts = [(int(p[0]*w_scale),int(p[1]*h_scale)) for p in best_person if p[2]>0.2]
			if len(valid_pts)>=3:
				x_coords, y_coords = [p[0] for p in valid_pts], [p[1] for p in valid_pts]
				min_x,max_x,min_y,max_y = min(x_coords),max(x_coords),min(y_coords),max(y_coords)
				pad_x,pad_y = (max_x-min_x)*0.3, (max_y-min_y)*0.3
				cv2.rectangle(mask,(max(0,int(min_x-pad_x)),max(0,int(min_y-pad_y))),(min(self.target_size[0],int(max_x+pad_x)),min(self.target_size[1],int(max_y+pad_y))),128,-1)
			for idx in [0,1,2,3,4]:
				if idx < len(best_person) and best_person[idx][2]>0.3: cv2.circle(mask,(int(best_person[idx][0]*w_scale),int(best_person[idx][1]*h_scale)),max(120,self.target_size[0]//12),255,-1)
			kernel_size = max(151,self.target_size[0]//12); kernel_size += (kernel_size%2==0)
			mask = cv2.GaussianBlur(mask, (kernel_size,kernel_size),0); mask = cv2.convertScaleAbs(mask,alpha=1.8,beta=0)
		except Exception as e: print(f"Error in pose mask: {e}"); return self.create_edge_detail_mask()
		return mask
	def create_edge_detail_mask(self) -> np.ndarray:
		mask = np.zeros(self.target_size[::-1], dtype=np.uint8)
		if self.target_image is not None:
			gray = cv2.cvtColor(self.target_image, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 100, 200)
			kernel_size = max(101,self.target_size[0]//15); kernel_size += (kernel_size%2==0)
			blur = cv2.GaussianBlur(edges, (kernel_size,kernel_size), 0)
			mask = cv2.normalize(blur,None,0,255,cv2.NORM_MINMAX).astype(np.uint8); mask = cv2.convertScaleAbs(mask,alpha=1.5,beta=0)
		return mask
	def setup_regions(self, pose_keypoints:Optional[np.ndarray]):
		self.detail_mask, self.regions = self.create_pose_detail_mask(pose_keypoints), []
		for scale in self.scales:
			for i in range(math.ceil(self.target_size[1]/scale)):
				for j in range(math.ceil(self.target_size[0]/scale)):
					y1,x1,y2,x2 = i*scale, j*scale, min((i+1)*scale,self.target_size[1]), min((j+1)*scale,self.target_size[0])
					chosen_scale = self.scales[-1] if self.detail_mask[y1:y2,x1:x2].mean()>=self.scale_thresholds[1] else (self.scales[1] if self.detail_mask[y1:y2,x1:x2].mean()>=self.scale_thresholds[0] else self.scales[0])
					if chosen_scale == scale: self.regions.append((i,j,scale))
		self.genome = [random.randrange(self.tile_library.num_variants) for _ in self.regions]; print(f"Setup {len(self.regions)} regions")
	def render_mosaic(self) -> np.ndarray:
		mosaic = np.ones(self.target_size[::-1], dtype=np.uint8)*255
		for (i,j,scale),variant_idx in zip(self.regions, self.genome):
			y1,x1,y2,x2 = i*scale, j*scale, min((i+1)*scale,self.target_size[1]), min((j+1)*scale,self.target_size[0])
			tile = self.tile_library.render_variant(variant_idx, scale)
			mosaic[y1:min(y1+tile.shape[0],y2), x1:min(x1+tile.shape[1],x2)] = tile[:y2-y1, :x2-x1]
		return mosaic
	def calculate_score(self) -> float:
		if self.target_image is None or self.mosaic is None: return float('inf')
		return np.mean((self.mosaic.astype(float)-cv2.cvtColor(self.target_image,cv2.COLOR_BGR2GRAY).astype(float))**2)
	def initialize(self, target_image:np.ndarray, pose_keypoints:Optional[np.ndarray], WINDOW_SIZE_H_W:Tuple[int,int]):
		self.target_image, self.WINDOW_SIZE_H_W = target_image, WINDOW_SIZE_H_W
		self.setup_regions(pose_keypoints); self.mosaic = self.render_mosaic()
		self.best_score, self.current_iteration = self.calculate_score(), 0; print(f"Initial MSE: {self.best_score:.2f}")
	def evolve_step(self) -> bool:
		if not self.regions or self.current_iteration >= self.max_iterations: return False
		region_idx = random.randrange(len(self.regions))
		old_variant,new_variant = self.genome[region_idx], random.randrange(self.tile_library.num_variants)
		if new_variant != old_variant:
			i,j,scale = self.regions[region_idx]
			y1,x1,y2,x2 = i*scale, j*scale, min((i+1)*scale,self.target_size[1]), min((j+1)*scale,self.target_size[0])
			target_region = cv2.cvtColor(self.target_image,cv2.COLOR_BGR2GRAY)[y1:y2,x1:x2]
			old_tile, new_tile = self.tile_library.render_variant(old_variant,scale)[:y2-y1,:x2-x1], self.tile_library.render_variant(new_variant,scale)[:y2-y1,:x2-x1]
			old_error = np.mean((old_tile.astype(float)-target_region.astype(float))**2)
			new_error = np.mean((new_tile.astype(float)-target_region.astype(float))**2)
			if new_error < old_error:
				self.genome[region_idx], self.best_score = new_variant, self.best_score + (new_error-old_error)
				self.mosaic[y1:y2, x1:x2] = new_tile; self.current_iteration += 1; return True
		self.current_iteration += 1; return False

class GeneticTileArtApp:
	def __init__(self):
		self.fullhd_width, self.fullhd_height = 1920, 1080
		self.rotate_90_degrees = True

		self.display_width = self.fullhd_height if self.rotate_90_degrees else self.fullhd_width
		self.display_height = self.fullhd_width if self.rotate_90_degrees else self.fullhd_height
		self.crop_width, self.crop_height = 1064, 1824
		self.margin_left, self.margin_top = 8, 8
		self.margin_right = self.display_width - self.crop_width - self.margin_left
		self.margin_bottom = self.display_height - self.crop_height - self.margin_top

		pygame.init()
		self.screen = pygame.display.set_mode((self.display_width, self.display_height), pygame.FULLSCREEN)
		pygame.display.set_caption("Genetic Tile Art")

		self.black, self.white, self.green, self.red, self.yellow = (0,0,0), (255,255,255), (0,255,0), (255,0,0), (255,255,0)
		self.debug_modes = ["OFF", "POSE", "POSTERIZED", "MASK"]
		self.current_debug_mode = 0
		self.crop_modes = ["FULL_POSE", "FACE_ONLY", "HEAD_SHOULDERS"]
		self.current_crop_mode = 1

		self.visibility_threshold, self.detection_threshold, self.crop_padding_factor = 0.3, 0.3, 0.2
		self.last_keypoints, self.last_scores, self.pose_detected = None, None, False

		self.camera_resolution = (2028, 1520)
		self.WINDOW_SIZE_H_W = (1520, 2028)

		self.tile_library = TileLibrary("tiles")
		self.genetic_generator = GeneticMosaicGenerator(self.tile_library, (self.crop_width, self.crop_height))
		self.evolution_active, self.evolution_paused, self.steps_per_frame = False, False, 10

		self.current_frame, self.frozen_frame, self.frozen_posterized, self.frozen_keypoints = None, None, None, None
		self.current_frame_crop_region = None
		self.frame_times, self.current_fps = [], 0.0

		self.setup_imx500_camera()

		self.font_small, self.font_medium, self.font_large = pygame.font.Font(None, 24), pygame.font.Font(None, 32), pygame.font.Font(None, 48)
		self.clock = pygame.time.Clock()
		print("Portrait HD Genetic Tile Art Started")
		print(f"Controls: SPACE(Evolve), P(Pause), R(Reset), D(Debug), M(Crop), S(Save), ESC(Exit)")

	def setup_imx500_camera(self):
		try:
			self.imx500 = IMX500("/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
			self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
			self.intrinsics.task, self.intrinsics.inference_rate = "pose estimation", self.intrinsics.inference_rate or 15
			self.intrinsics.update_with_defaults()
			self.picam2 = Picamera2(self.imx500.camera_num)
			config = self.picam2.create_preview_configuration(main={"size": self.camera_resolution, "format": "BGR888"},
				controls={'FrameRate': self.intrinsics.inference_rate}, buffer_count=12)
			self.picam2.configure(config)
			self.picam2.pre_callback = self.ai_output_tensor_parse
			self.imx500.show_network_fw_progress_bar()
			self.picam2.start(config, show_preview=False); self.imx500.set_auto_aspect_ratio()
			print("IMX500 pose estimation initialized")
		except Exception as e:
			print(f"Failed to init IMX500: {e}. Falling back."); self.imx500 = None
			self.picam2 = Picamera2()
			self.picam2.configure(self.picam2.create_preview_configuration(main={"size": self.camera_resolution, "format": "BGR888"}))
			self.picam2.start()

	def ai_output_tensor_parse(self, request: CompletedRequest):
		try:
			with MappedArray(request, "main") as m: self.current_frame = m.array
		except Exception: return
		self.pose_detected = False
		if self.imx500:
			try:
				outputs = self.imx500.get_outputs(metadata=request.get_metadata(), add_batch=True)
				if outputs is not None:
					kps, scores, _ = postprocess_higherhrnet(outputs=outputs, img_size=self.WINDOW_SIZE_H_W,
						img_w_pad=(0,0), img_h_pad=(0,0), detection_threshold=self.detection_threshold, network_postprocess=True)
					if scores:
						self.last_keypoints = np.reshape(np.stack(kps, axis=0), (len(scores), 17, 3))
						self.last_scores, self.pose_detected = scores, True
			except Exception as e: print(f"Error parsing pose: {e}")
		self.current_frame_crop_region = self._calculate_crop_region() if self.pose_detected else None

	def _calculate_crop_region(self):
		if self.last_keypoints is None: return None
		try:
			person_kps = self.last_keypoints[np.argmax(self.last_scores)]
			mode_indices = {"FACE_ONLY":[0,1,2,3,4], "HEAD_SHOULDERS":[0,1,2,3,4,5,6]}.get(self.crop_modes[self.current_crop_mode], list(range(17)))
			valid_pts = [(kp[0], kp[1]) for i, kp in enumerate(person_kps) if i in mode_indices and kp[2] > self.visibility_threshold]
			if len(valid_pts) < 2: return None
			x_coords, y_coords = [p[0] for p in valid_pts], [p[1] for p in valid_pts]
			min_x, max_x, min_y, max_y = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
			w, h = max_x-min_x, max_y-min_y
			pad_x, pad_y = w*self.crop_padding_factor, h*self.crop_padding_factor
			crop_x, crop_y = max(0, min_x-pad_x), max(0, min_y-pad_y)
			crop_w, crop_h = min(self.camera_resolution[0]-crop_x, w+2*pad_x), min(self.camera_resolution[1]-crop_y, h+2*pad_y)
			content_aspect, crop_aspect = self.crop_width/self.crop_height, crop_w/crop_h
			if crop_aspect > content_aspect: crop_h = crop_w / content_aspect
			else: crop_w = crop_h * content_aspect
			crop_x, crop_y = max(0, min_x + w/2 - crop_w/2), max(0, min_y + h/2 - crop_h/2)
			crop_w, crop_h = min(crop_w, self.camera_resolution[0]-crop_x), min(crop_h, self.camera_resolution[1]-crop_y)
			return (int(crop_x), int(crop_y), int(crop_w), int(crop_h))
		except Exception: return None

	def freeze_and_start_evolution(self):
		if not self.pose_detected or self.current_frame is None or self.current_frame_crop_region is None:
			print("No pose detected, cannot freeze"); return
		try:
			x,y,w,h = self.current_frame_crop_region
			self.frozen_frame = self.current_frame[y:y+h, x:x+w].copy()
			self.frozen_posterized = self.posterize_frame(self.frozen_frame)
			self.frozen_keypoints = self.last_keypoints[np.argmax(self.last_scores)].copy()
			target_image = cv2.resize(self.frozen_frame, (self.crop_width, self.crop_height))
			self.genetic_generator.initialize(target_image, self.frozen_keypoints, self.WINDOW_SIZE_H_W)
			self.evolution_active, self.evolution_paused = True, False
			print(f"Frame frozen! Starting evolution...")
		except Exception as e: print(f"Error freezing: {e}")

	def posterize_frame(self, frame: np.ndarray) -> np.ndarray:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		_, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
		return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

	def handle_events(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): return False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE: self.freeze_and_start_evolution()
				elif event.key == pygame.K_p and self.evolution_active: self.evolution_paused = not self.evolution_paused; print(f"Evo {'paused' if self.evolution_paused else 'resumed'}")
				elif event.key == pygame.K_r: self.evolution_active, self.evolution_paused = False, False; print("Evolution reset")
				elif event.key == pygame.K_d: self.current_debug_mode = (self.current_debug_mode + 1) % len(self.debug_modes); print(f"Debug: {self.debug_modes[self.current_debug_mode]}")
				elif event.key == pygame.K_m: self.current_crop_mode = (self.current_crop_mode + 1) % len(self.crop_modes); print(f"Crop: {self.crop_modes[self.current_crop_mode]}")
				elif event.key == pygame.K_s and self.genetic_generator.mosaic is not None:
					filename = f"mosaic_{self.genetic_generator.current_iteration}.png"
					cv2.imwrite(filename, self.genetic_generator.mosaic); print(f"Saved {filename}")
		return True

	def _prepare_bgr_surface_for_display(self, bgr_array: np.ndarray, target_size: Tuple[int, int]) -> pygame.Surface:
		"""
		FIXED: Central function to correctly orient, scale, and color-convert a BGR numpy array for display.
		"""
		# 1. Rotate the image array using OpenCV if required.
		if self.rotate_90_degrees:
			bgr_array = cv2.rotate(bgr_array, cv2.ROTATE_90_CLOCKWISE)

		# 2. Scale the (now correctly oriented) array to the final display size.
		final_array = cv2.resize(bgr_array, target_size, interpolation=cv2.INTER_AREA)

		# 3. Convert from BGR to RGB for Pygame's make_surface.
		rgb_array = cv2.cvtColor(final_array, cv2.COLOR_BGR2RGB)

		# 4. Transpose and create the surface.
		return pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2)))

	def draw_frame(self):
		self.screen.fill(self.black)
		main_rect = pygame.Rect(self.margin_left, self.margin_top, self.crop_width, self.crop_height)

		if self.evolution_active and self.genetic_generator.mosaic is not None:
			mosaic_bgr = cv2.cvtColor(self.genetic_generator.mosaic, cv2.COLOR_GRAY2BGR)
			surface = self._prepare_bgr_surface_for_display(mosaic_bgr, main_rect.size)
			self.screen.blit(surface, main_rect.topleft)
		elif self.current_frame is not None and self.current_frame_crop_region is not None:
			x, y, w, h = self.current_frame_crop_region
			crop_region = self.current_frame[y:y+h, x:x+w]
			if crop_region.size > 0:
				surface = self._prepare_bgr_surface_for_display(crop_region, main_rect.size)
				self.screen.blit(surface, main_rect.topleft)
		else:
			loading_text = self.font_large.render("Waiting for pose...", True, self.white)
			self.screen.blit(loading_text, loading_text.get_rect(center=self.screen.get_rect().center))

		self.draw_debug_overlay()
		self.draw_info_overlay()

	def draw_debug_overlay(self):
		debug_mode = self.debug_modes[self.current_debug_mode]
		if debug_mode == "OFF": return

		target_rect = pygame.Rect(self.margin_left, self.margin_top, self.crop_width, self.crop_height)
		overlay_surface = None

		if debug_mode == "POSTERIZED" and self.evolution_active and self.frozen_posterized is not None:
			overlay_surface = self._prepare_bgr_surface_for_display(self.frozen_posterized, target_rect.size)
		elif debug_mode == "MASK" and self.genetic_generator.detail_mask is not None:
			mask_bgr = cv2.cvtColor(self.genetic_generator.detail_mask, cv2.COLOR_GRAY2BGR)
			overlay_surface = self._prepare_bgr_surface_for_display(mask_bgr, target_rect.size)

		if overlay_surface:
			self.screen.blit(overlay_surface, target_rect.topleft)

		# Pose must be drawn on top of everything else
		if debug_mode == "POSE":
			self.draw_pose_overlay()

	def draw_pose_overlay(self):
		kps_source = self.frozen_keypoints if self.evolution_active and self.frozen_keypoints is not None else \
			 (self.last_keypoints[np.argmax(self.last_scores)] if self.pose_detected and self.last_scores is not None else None)
		crop_region = self.current_frame_crop_region
		if kps_source is None or crop_region is None: return

		screen_points = {}
		crop_x, crop_y, crop_w, crop_h = crop_region
		for idx, (kp_x, kp_y, vis) in enumerate(kps_source):
			if vis > self.visibility_threshold:
				rel_x, rel_y = (kp_x - crop_x) / crop_w, (kp_y - crop_y) / crop_h
				if self.rotate_90_degrees:
					# Transform for 90-degree clockwise rotation: (rel_x, rel_y) -> (1-rel_y, rel_x)
					screen_x = self.margin_left + (1 - rel_y) * self.crop_width
					screen_y = self.margin_top + rel_x * self.crop_height
				else:
					screen_x = self.margin_left + rel_x * self.crop_width
					screen_y = self.margin_top + rel_y * self.crop_height
				screen_points[idx] = (int(screen_x), int(screen_y))

		skeleton = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),(5,7),(6,8),(7,9),(8,10),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]
		for start, end in skeleton:
			if start in screen_points and end in screen_points:
				pygame.draw.line(self.screen, self.white, screen_points[start], screen_points[end], 3)
		for idx, pos in screen_points.items():
			color = self.red if idx==0 else (self.yellow if idx in [1,2,3,4] else (self.green if idx in [5,6] else self.white))
			pygame.draw.circle(self.screen, color, pos, 8)

	def draw_info_overlay(self):
		self.frame_times.append(time.time()); self.frame_times = self.frame_times[-30:]
		if len(self.frame_times) > 1: self.current_fps = (len(self.frame_times)-1) / (self.frame_times[-1]-self.frame_times[0])

		info_y = self.display_height - self.margin_bottom + 15
		status = "EVOLVING" if self.evolution_active and not self.evolution_paused else ("PAUSED" if self.evolution_paused else "WAITING")
		status_color = self.green if status=="EVOLVING" else (self.yellow if status=="PAUSED" else self.red)
		self.screen.blit(self.font_medium.render(f"Status: {status}", True, status_color), (self.margin_left+10, info_y))
		if self.evolution_active:
			self.screen.blit(self.font_medium.render(f"Iter: {self.genetic_generator.current_iteration}", True, self.white), (self.margin_left+250, info_y))
			self.screen.blit(self.font_medium.render(f"MSE: {self.genetic_generator.best_score:.1f}", True, self.white), (self.margin_left+450, info_y))
		fps_text = self.font_medium.render(f"FPS: {self.current_fps:.1f}", True, self.white)
		self.screen.blit(fps_text, fps_text.get_rect(right=self.display_width-self.margin_right-10, centery=info_y+fps_text.get_height()//2))

	def run(self):
		running = True
		while running:
			running = self.handle_events()
			if self.evolution_active and not self.evolution_paused:
				for _ in range(self.steps_per_frame):
					if self.genetic_generator.evolve_step() and self.genetic_generator.current_iteration % 500 == 0:
						print(f"Iter {self.genetic_generator.current_iteration}, MSE: {self.genetic_generator.best_score:.2f}")
			self.draw_frame()
			pygame.display.flip()
			self.clock.tick(30)
		if self.picam2: self.picam2.stop()
		pygame.quit()

if __name__ == "__main__":
	try:
		app = GeneticTileArtApp()
		app.run()
	except KeyboardInterrupt:
		print("\nExiting...")
	except Exception as e:
		print(f"\nAn unexpected error occurred: {e}")
		import traceback; traceback.print_exc()