#!/usr/bin/env python3
"""
IMX500 Genetic Tile Art - Portrait HD Display
Full-size crop/transformation for rotated HD TV in portrait orientation
Target size: 1064x1824 with proper margins

Key Features:
- Portrait HD display (1080x1920) with exact margins (8,8,88,8)
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

class TileLibrary:
	"""
	Manages B&W tile images and their rotations

	Loads PNG tiles from 'tiles' directory, converts to B&W, creates 4 rotations
	Adds empty (white) and full (black) tiles programmatically
	"""

	def __init__(self, tiles_dir: str = "tiles"):
		self.tiles_dir = tiles_dir
		self.tile_variants = []  # All tiles with all rotations
		self.num_variants = 0
		self.base_tile_size = 20  # Will be updated from loaded tiles

		self.load_tiles()

	def load_tiles(self):
		"""Load PNG files, convert to B&W, create rotations"""
		tile_paths = glob.glob(os.path.join(self.tiles_dir, "*.png"))

		if not tile_paths:
			print(f"Warning: No PNG files found in {self.tiles_dir}, creating basic tiles")
			self.create_basic_tiles()
			return

		base_tiles = []

		# Load all PNG files and convert to B&W
		for path in sorted(tile_paths):
			try:
				img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
				if img is None:
					continue

				# Handle different channel formats (grayscale, RGB, RGBA)
				if img.ndim == 2:  # Grayscale
					b = g = r = img
					a = np.full_like(img, 255)
					img = cv2.merge([b, g, r, a])
				elif img.shape[2] == 3:  # RGB
					b, g, r = cv2.split(img)
					a = np.full_like(b, 255)
					img = cv2.merge([b, g, r, a])

				# Composite onto white background using alpha channel
				rgb = img[:, :, :3].astype(float)
				alpha = img[:, :, 3:4].astype(float) / 255.0
				composited = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)

				# Convert to grayscale and then to pure B&W
				gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
				_, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

				base_tiles.append(bw)
				print(f"Loaded {path}")

			except Exception as e:
				print(f"Error loading {path}: {e}")

		if not base_tiles:
			print("No tiles loaded, creating basic tiles")
			self.create_basic_tiles()
			return

		# Set tile size from first loaded tile
		self.base_tile_size = base_tiles[0].shape[0]

		# Add programmatic empty (all white) and full (all black) tiles
		empty_tile = np.ones((self.base_tile_size, self.base_tile_size), dtype=np.uint8) * 255
		full_tile = np.zeros((self.base_tile_size, self.base_tile_size), dtype=np.uint8)
		base_tiles.extend([empty_tile, full_tile])

		# Create all rotation variants (0°, 90°, 180°, 270°)
		self.tile_variants = []
		for tile in base_tiles:
			for angle in [0, 90, 180, 270]:
				center = (tile.shape[1] / 2, tile.shape[0] / 2)
				M = cv2.getRotationMatrix2D(center, angle, 1)
				rotated = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]),
									   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
				self.tile_variants.append(rotated)

		self.num_variants = len(self.tile_variants)
		print(f"Created {self.num_variants} B&W tile variants")

	def create_basic_tiles(self):
		"""Create basic geometric B&W tiles if no PNGs found"""
		self.base_tile_size = 20
		size = self.base_tile_size

		# Create simple geometric shapes
		shapes = []

		# Diamond
		diamond = np.ones((size, size), dtype=np.uint8) * 255
		points = np.array([[size//2, 2], [size-2, size//2], [size//2, size-2], [2, size//2]])
		cv2.fillPoly(diamond, [points], 0)
		shapes.append(diamond)

		# Triangle
		triangle = np.ones((size, size), dtype=np.uint8) * 255
		points = np.array([[size//2, 2], [size-2, size-2], [2, size-2]])
		cv2.fillPoly(triangle, [points], 0)
		shapes.append(triangle)

		# Circle
		circle = np.ones((size, size), dtype=np.uint8) * 255
		cv2.circle(circle, (size//2, size//2), size//2 - 2, 0, -1)
		shapes.append(circle)

		# Cross
		cross = np.ones((size, size), dtype=np.uint8) * 255
		thickness = 4
		cv2.rectangle(cross, (size//2 - thickness//2, 2), (size//2 + thickness//2, size-2), 0, -1)
		cv2.rectangle(cross, (2, size//2 - thickness//2), (size-2, size//2 + thickness//2), 0, -1)
		shapes.append(cross)

		# Empty and full tiles
		empty = np.ones((size, size), dtype=np.uint8) * 255
		full = np.zeros((size, size), dtype=np.uint8)
		shapes.extend([empty, full])

		# Create rotations for all shapes
		self.tile_variants = []
		for tile in shapes:
			for angle in [0, 90, 180, 270]:
				center = (tile.shape[1] / 2, tile.shape[0] / 2)
				M = cv2.getRotationMatrix2D(center, angle, 1)
				rotated = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]),
									   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
				self.tile_variants.append(rotated)

		self.num_variants = len(self.tile_variants)
		print(f"Created {self.num_variants} basic B&W tile variants")

	def render_variant(self, variant_idx: int, target_size: int) -> np.ndarray:
		"""Render a specific tile variant at the requested size"""
		if variant_idx >= self.num_variants:
			variant_idx = variant_idx % self.num_variants

		tile = self.tile_variants[variant_idx]
		if target_size != self.base_tile_size:
			return cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_AREA)
		return tile.copy()

class GeneticMosaicGenerator:
	"""
	Advanced genetic algorithm for B&W tile mosaic generation

	Uses pose-based detail mask to determine tile sizes:
	- Face areas: 38px tiles (high detail)
	- Shoulder areas: 76px tiles (medium detail)
	- Background: 152px tiles (low detail)

	Genetic algorithm mutates individual tiles and accepts improvements
	"""

	def __init__(self, tile_library: TileLibrary, target_size: Tuple[int, int]):
		self.tile_library = tile_library
		self.target_size = target_size  # (width, height)
		self.scales = [152, 76, 38]  # Large, medium, small tiles
		self.scale_thresholds = [80, 180]  # Detail mask brightness thresholds

		# Genetic algorithm state
		self.regions = []  # List of (row, col, scale) for each tile
		self.genome = []   # List of tile variant indices
		self.mosaic = None # Current rendered mosaic
		self.target_image = None # B&W target to match
		self.detail_mask = None  # Mask determining tile sizes
		self.best_score = float('inf')

		# Evolution parameters
		self.max_iterations = 30000
		self.current_iteration = 0

	def create_pose_detail_mask(self, pose_keypoints: Optional[np.ndarray]) -> np.ndarray:
		"""
		Create detail mask where face/shoulders get high detail (small tiles)

		Uses pose keypoints to create circular regions:
		- Face keypoints (nose, eyes, ears) -> high detail (white in mask)
		- Shoulder keypoints -> medium detail (gray in mask)
		- Background -> low detail (black in mask)
		"""
		mask = np.zeros(self.target_size[::-1], dtype=np.uint8)  # Height x Width

		if pose_keypoints is not None and len(pose_keypoints) > 0:
			try:
				# Handle keypoint format (single person or multiple people)
				if pose_keypoints.ndim == 3:
					best_person = pose_keypoints[0]
				else:
					best_person = pose_keypoints

				if best_person.shape[0] < 17 or best_person.shape[1] != 3:
					return self.create_edge_detail_mask()

				# COCO keypoint indices
				face_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
				shoulder_indices = [5, 6]        # shoulders

				# Scale keypoints from camera space to target size
				# Fixed coordinate scaling using target size directly
				h_scale = self.target_size[1] / self.WINDOW_SIZE_H_W[0]  # Use actual camera height
				w_scale = self.target_size[0] / self.WINDOW_SIZE_H_W[1]  # Use actual camera width

				# Create high-detail regions around face (larger areas)
				for idx in face_indices:
					if idx < len(best_person):
						keypoint = best_person[idx]
						if len(keypoint) >= 3:
							x, y, confidence = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
							if confidence > 0.3:
								scaled_x = int(x * w_scale)
								scaled_y = int(y * h_scale)
								radius = max(120, self.target_size[0] // 12)  # Larger radius for face features
								cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)

				# Create medium-detail regions around shoulders and body
				for idx in shoulder_indices:
					if idx < len(best_person):
						keypoint = best_person[idx]
						if len(keypoint) >= 3:
							x, y, confidence = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
							if confidence > 0.3:
								scaled_x = int(x * w_scale)
								scaled_y = int(y * h_scale)
								radius = max(100, self.target_size[0] // 15)  # Large radius for shoulders
								cv2.circle(mask, (scaled_x, scaled_y), radius, 128, -1)

				# Add medium detail for the whole person area
				all_valid_points = []
				for idx in range(17):  # All COCO keypoints
					if idx < len(best_person):
						keypoint = best_person[idx]
						if len(keypoint) >= 3:
							x, y, confidence = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
							if confidence > 0.2:  # Lower threshold for general person area
								scaled_x = int(x * w_scale)
								scaled_y = int(y * h_scale)
								all_valid_points.append((scaled_x, scaled_y))

				# Create bounding box around all person keypoints and fill with medium detail
				if len(all_valid_points) >= 3:
					x_coords = [p[0] for p in all_valid_points]
					y_coords = [p[1] for p in all_valid_points]
					min_x, max_x = min(x_coords), max(x_coords)
					min_y, max_y = min(y_coords), max(y_coords)

					# Expand bounding box
					padding_x = (max_x - min_x) * 0.3
					padding_y = (max_y - min_y) * 0.3

					expanded_min_x = max(0, int(min_x - padding_x))
					expanded_max_x = min(self.target_size[0], int(max_x + padding_x))
					expanded_min_y = max(0, int(min_y - padding_y))
					expanded_max_y = min(self.target_size[1], int(max_y + padding_y))

					# Fill person area with medium detail (gray)
					cv2.rectangle(mask, (expanded_min_x, expanded_min_y),
								(expanded_max_x, expanded_max_y), 128, -1)

				# Re-draw face areas on top to ensure they stay high detail
				for idx in face_indices:
					if idx < len(best_person):
						keypoint = best_person[idx]
						if len(keypoint) >= 3:
							x, y, confidence = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
							if confidence > 0.3:
								scaled_x = int(x * w_scale)
								scaled_y = int(y * h_scale)
								radius = max(120, self.target_size[0] // 12)
								cv2.circle(mask, (scaled_x, scaled_y), radius, 255, -1)

				# Blur mask for smooth transitions between detail levels
				kernel_size = max(151, self.target_size[0] // 12)  # Larger kernel for better blending
				if kernel_size % 2 == 0:
					kernel_size += 1
				mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
				mask = cv2.convertScaleAbs(mask, alpha=1.8, beta=0)  # Higher contrast for better separation

			except Exception as e:
				print(f"Error creating pose detail mask: {e}")
				return self.create_edge_detail_mask()
		else:
			return self.create_edge_detail_mask()

		return mask

	def create_edge_detail_mask(self) -> np.ndarray:
		"""Create detail mask based on edge detection when no pose available"""
		mask = np.zeros(self.target_size[::-1], dtype=np.uint8)

		if self.target_image is not None:
			gray = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)
			edges = cv2.Canny(gray, 100, 200)
			kernel_size = max(101, self.target_size[0] // 15)
			if kernel_size % 2 == 0:
				kernel_size += 1
			blur = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)
			mask = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
			mask = cv2.convertScaleAbs(mask, alpha=1.5, beta=0)

		return mask

	def setup_regions(self, pose_keypoints: Optional[np.ndarray]):
		"""
		Setup grid regions based on detail mask

		Creates a grid of tiles with different sizes based on detail mask:
		- Bright areas in mask -> small tiles (38px)
		- Medium areas -> medium tiles (76px)
		- Dark areas -> large tiles (152px)
		"""
		self.detail_mask = self.create_pose_detail_mask(pose_keypoints)
		self.regions = []

		# Process each tile scale
		for scale in self.scales:
			grid_h = math.ceil(self.target_size[1] / scale)
			grid_w = math.ceil(self.target_size[0] / scale)

			# Check each grid cell
			for i in range(grid_h):
				for j in range(grid_w):
					y1, x1 = i * scale, j * scale
					y2 = min(y1 + scale, self.target_size[1])
					x2 = min(x1 + scale, self.target_size[0])

					# Calculate average detail mask value for this region
					mask_region = self.detail_mask[y1:y2, x1:x2]
					avg_detail = mask_region.mean()

					# Choose scale based on detail level
					if avg_detail >= self.scale_thresholds[1]:
						chosen_scale = self.scales[-1]  # 38px (highest detail)
					elif avg_detail >= self.scale_thresholds[0]:
						chosen_scale = self.scales[1]   # 76px (medium detail)
					else:
						chosen_scale = self.scales[0]   # 152px (lowest detail)

					# Only add this region if it matches the current scale
					if chosen_scale == scale:
						self.regions.append((i, j, scale))

		# Initialize random genome (one tile variant index per region)
		self.genome = [random.randrange(self.tile_library.num_variants) for _ in self.regions]
		print(f"Setup {len(self.regions)} regions with {len(set(self.genome))} distinct variants")

	def render_mosaic(self) -> np.ndarray:
		"""Render current genome to B&W mosaic image"""
		mosaic = np.ones((self.target_size[1], self.target_size[0]), dtype=np.uint8) * 255

		# Place each tile according to genome
		for (i, j, scale), variant_idx in zip(self.regions, self.genome):
			y1, x1 = i * scale, j * scale
			y2 = min(y1 + scale, self.target_size[1])
			x2 = min(x1 + scale, self.target_size[0])

			tile = self.tile_library.render_variant(variant_idx, scale)

			# Handle edge cases where tile might be larger than remaining space
			tile_h, tile_w = tile.shape
			region_h, region_w = y2 - y1, x2 - x1

			if tile_h > region_h or tile_w > region_w:
				tile = tile[:region_h, :region_w]

			mosaic[y1:y2, x1:x2] = tile

		return mosaic

	def calculate_score(self) -> float:
		"""Calculate Mean Squared Error between mosaic and target"""
		if self.target_image is None or self.mosaic is None:
			return float('inf')

		target_gray = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)
		return np.mean((self.mosaic.astype(float) - target_gray.astype(float)) ** 2)

	def initialize(self, target_image: np.ndarray, pose_keypoints: Optional[np.ndarray]):
		"""Initialize genetic algorithm with target image and pose"""
		self.target_image = target_image
		self.setup_regions(pose_keypoints)
		self.mosaic = self.render_mosaic()
		self.best_score = self.calculate_score()
		self.current_iteration = 0
		print(f"Initial MSE: {self.best_score:.2f}")

	def evolve_step(self) -> bool:
		"""
		Perform one evolution step

		Genetic algorithm approach:
		1. Select random tile region
		2. Try random new tile variant
		3. Calculate local fitness improvement
		4. Accept if better, reject if worse

		This is more efficient than evaluating whole population
		"""
		if not self.regions or self.current_iteration >= self.max_iterations:
			return False

		# Select random region to mutate
		region_idx = random.randrange(len(self.regions))
		old_variant = self.genome[region_idx]
		new_variant = random.randrange(self.tile_library.num_variants)

		if new_variant != old_variant:
			i, j, scale = self.regions[region_idx]
			y1, x1 = i * scale, j * scale
			y2 = min(y1 + scale, self.target_size[1])
			x2 = min(x1 + scale, self.target_size[0])

			# Get target region and tile variants
			target_region = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)[y1:y2, x1:x2]
			old_tile = self.tile_library.render_variant(old_variant, scale)
			new_tile = self.tile_library.render_variant(new_variant, scale)

			# Handle size mismatches
			region_h, region_w = target_region.shape
			if old_tile.shape[0] > region_h or old_tile.shape[1] > region_w:
				old_tile = old_tile[:region_h, :region_w]
			if new_tile.shape[0] > region_h or new_tile.shape[1] > region_w:
				new_tile = new_tile[:region_h, :region_w]

			# Calculate local error for both tiles
			old_error = np.mean((old_tile.astype(float) - target_region.astype(float)) ** 2)
			new_error = np.mean((new_tile.astype(float) - target_region.astype(float)) ** 2)

			# Accept improvement
			if new_error < old_error:
				self.genome[region_idx] = new_variant
				self.mosaic[y1:y2, x1:x2] = new_tile
				self.best_score += (new_error - old_error)
				self.current_iteration += 1
				return True

		self.current_iteration += 1
		return False

class GeneticTileArtApp:
	"""
	Main application for portrait HD genetic tile art

	Features:
	- Portrait HD display (1080x1920) for rotated TV
	- Exact margins: 8px left/top, 88px right, 8px bottom
	- Real-time pose detection and crop calculation
	- Debug overlays: pose, posterized target, detail mask
	- Genetic evolution of tile mosaics
	"""

	def __init__(self):
		# TV is actually 1920x1080 (landscape) but rotated 90° for portrait viewing
		self.fullhd_width = 1920
		self.fullhd_height = 1080
		self.rotate_90_degrees = True  # Rotate for portrait TV orientation

		# If rotated, swap width/height for display
		if self.rotate_90_degrees:
			self.display_width = self.fullhd_height   # 1080
			self.display_height = self.fullhd_width   # 1920
		else:
			self.display_width = self.fullhd_width    # 1920
			self.display_height = self.fullhd_height  # 1080

		# Target crop size and margins as specified
		self.margin_left = 8
		self.margin_top = 8
		self.margin_right = 88
		self.margin_bottom = 8
		self.crop_width = 1064   # 1080 - 8 - 8 = 1064
		self.crop_height = 1824  # 1920 - 8 - 88 = 1824

		# Initialize Pygame for fullscreen portrait display
		pygame.init()
		self.screen = pygame.display.set_mode((self.display_width, self.display_height), pygame.FULLSCREEN)
		pygame.display.set_caption("Genetic Tile Art - Portrait HD")

		# Colors
		self.black = (0, 0, 0)
		self.white = (255, 255, 255)
		self.green = (0, 255, 0)
		self.red = (255, 0, 0)
		self.yellow = (255, 255, 0)

		# Debug overlay modes (cycle with 'D' key)
		self.debug_modes = ["OFF", "POSE", "POSTERIZED", "MASK"]
		self.current_debug_mode = 0

		# Crop modes (cycle with 'M' key)
		self.crop_modes = ["FULL_POSE", "FACE_ONLY", "HEAD_SHOULDERS"]
		self.current_crop_mode = 1  # Start with FACE_ONLY

		# Pose detection parameters
		self.visibility_threshold = 0.3  # Minimum keypoint confidence
		self.detection_threshold = 0.3   # Minimum person confidence
		self.crop_padding_factor = 0.2   # Extra padding around detected area
		self.last_keypoints = None
		self.last_scores = None
		self.pose_detected = False

		# Fixed camera resolution (higher res from camera_tv_test.py)
		self.camera_resolution = (2028, 1520)  # Higher resolution for better quality
		self.WINDOW_SIZE_H_W = (1520, 2028)  # IMX500 pose model input size (H, W)

		# Load tile library
		self.tile_library = TileLibrary("tiles")

		# Genetic mosaic generator
		self.genetic_generator = GeneticMosaicGenerator(self.tile_library, (self.crop_width, self.crop_height))
		self.evolution_active = False
		self.evolution_paused = False
		self.steps_per_frame = 5  # Evolution steps per display frame

		# Camera frame storage
		self.current_frame = None
		self.frozen_frame = None
		self.frozen_posterized = None
		self.frozen_keypoints = None  # Store keypoints when frozen
		self.current_frame_crop_region = None

		# FPS tracking
		self.frame_times = []
		self.frame_count = 0
		self.current_fps = 0.0

		# Setup camera
		self.setup_imx500_camera()

		# UI fonts
		self.font_small = pygame.font.Font(None, 24)
		self.font_medium = pygame.font.Font(None, 32)
		self.font_large = pygame.font.Font(None, 48)
		self.clock = pygame.time.Clock()

		# Print startup info
		print("Portrait HD Genetic Tile Art Started")
		print(f"TV: {self.fullhd_width}x{self.fullhd_height} ({'rotated 90°' if self.rotate_90_degrees else 'normal'})")
		print(f"Display: {self.display_width}x{self.display_height}")
		print(f"Target crop: {self.crop_width}x{self.crop_height}")
		print(f"Camera: {self.camera_resolution[0]}x{self.camera_resolution[1]}")
		print(f"Tile library: {self.tile_library.num_variants} variants")
		print()
		print("Controls:")
		print("- SPACE: Freeze frame and start genetic evolution")
		print("- P: Pause/resume evolution")
		print("- R: Reset evolution")
		print("- D: Cycle debug overlay modes")
		print("- M: Cycle crop mode")
		print("- S: Save current mosaic")
		print("- ESC: Exit")

	def setup_imx500_camera(self):
		"""Initialize IMX500 camera with pose estimation"""
		try:
			model_path = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
			self.imx500 = IMX500(model_path)
			self.intrinsics = self.imx500.network_intrinsics

			if not self.intrinsics:
				self.intrinsics = NetworkIntrinsics()
				self.intrinsics.task = "pose estimation"

			if self.intrinsics.inference_rate is None:
				self.intrinsics.inference_rate = 15  # Higher FPS for smooth display

			self.intrinsics.update_with_defaults()

			self.picam2 = Picamera2(self.imx500.camera_num)
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"},
				controls={'FrameRate': self.intrinsics.inference_rate},
				buffer_count=12
			)

			self.picam2.configure(config)
			self.picam2.pre_callback = self.ai_output_tensor_parse

			self.imx500.show_network_fw_progress_bar()
			self.picam2.start(config, show_preview=False)
			self.imx500.set_auto_aspect_ratio()

			print("IMX500 pose estimation initialized")

		except Exception as e:
			print(f"Failed to initialize IMX500: {e}")
			# Fallback to regular camera
			self.picam2 = Picamera2()
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"}
			)
			self.picam2.configure(config)
			self.picam2.start()
			self.imx500 = None

	def update_fps(self):
		"""Update FPS calculation using rolling average"""
		current_time = time.time()
		self.frame_times.append(current_time)
		self.frame_count += 1

		if len(self.frame_times) > 30:
			self.frame_times.pop(0)

		if self.frame_count % 10 == 0 and len(self.frame_times) > 1:
			time_span = self.frame_times[-1] - self.frame_times[0]
			if time_span > 0:
				self.current_fps = (len(self.frame_times) - 1) / time_span

	def ai_output_tensor_parse(self, request: CompletedRequest):
		"""
		Parse pose estimation output from IMX500

		Called automatically for each camera frame
		Updates pose keypoints and calculates crop region
		"""
		try:
			with MappedArray(request, "main") as m:
				self.current_frame = m.array.copy()
		except:
			pass

		if not self.imx500:
			return

		try:
			np_outputs = self.imx500.get_outputs(metadata=request.get_metadata(), add_batch=True)

			if np_outputs is not None:
				keypoints, scores, boxes = postprocess_higherhrnet(
					outputs=np_outputs,
					img_size=self.WINDOW_SIZE_H_W,
					img_w_pad=(0, 0),
					img_h_pad=(0, 0),
					detection_threshold=self.detection_threshold,
					network_postprocess=True
				)

				if scores is not None and len(scores) > 0:
					self.last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
					self.last_scores = scores
					self.pose_detected = True
					self.current_frame_crop_region = self._calculate_crop_region()
				else:
					self.pose_detected = False
					self.current_frame_crop_region = None
			else:
				self.pose_detected = False
				self.current_frame_crop_region = None

		except Exception as e:
			print(f"Error parsing pose results: {e}")
			self.pose_detected = False
			self.current_frame_crop_region = None

	def _calculate_crop_region(self):
		"""
		Calculate crop region based on current crop mode
		Fixed coordinate calculations matching camera_tv_test.py properly

		Uses detected pose keypoints to determine crop area:
		- FACE_ONLY: nose, eyes, ears
		- HEAD_SHOULDERS: face + shoulders
		- FULL_POSE: all visible keypoints

		Maintains target aspect ratio (1064:1824) for rotated display
		"""
		if not self.pose_detected or self.last_keypoints is None:
			return None

		try:
			# Handle different score formats - find best person
			if isinstance(self.last_scores, (list, np.ndarray)) and len(self.last_scores) > 0:
				if len(self.last_scores) == 1:
					best_person_idx = 0
				else:
					best_person_idx = np.argmax(self.last_scores)
			else:
				best_person_idx = 0

			if best_person_idx >= len(self.last_keypoints):
				return None

			person_keypoints = self.last_keypoints[best_person_idx]
			if len(person_keypoints) < 17:
				return None

			# Select keypoints based on crop mode
			crop_mode = self.crop_modes[self.current_crop_mode]

			if crop_mode == "FACE_ONLY":
				target_indices = [0, 1, 2, 3, 4]  # face keypoints
			elif crop_mode == "HEAD_SHOULDERS":
				target_indices = [0, 1, 2, 3, 4, 5, 6]  # face + shoulders
			else:  # FULL_POSE
				target_indices = list(range(17))  # all keypoints

			# Collect valid keypoints above visibility threshold
			valid_points = []
			for idx in target_indices:
				if idx < len(person_keypoints):
					kp_x, kp_y, visibility = person_keypoints[idx]
					if visibility > self.visibility_threshold:
						valid_points.append((kp_x, kp_y))

			if len(valid_points) < 2:
				return None

			# Calculate bounding box of valid keypoints
			x_coords = [p[0] for p in valid_points]
			y_coords = [p[1] for p in valid_points]

			min_x, max_x = min(x_coords), max(x_coords)
			min_y, max_y = min(y_coords), max(y_coords)

			width = max_x - min_x
			height = max_y - min_y

			# Add padding around detected area
			padding_x = width * self.crop_padding_factor
			padding_y = height * self.crop_padding_factor

			# Calculate initial crop region
			crop_x = max(0, min_x - padding_x)
			crop_y = max(0, min_y - padding_y)
			crop_w = min(self.camera_resolution[0] - crop_x, width + 2 * padding_x)
			crop_h = min(self.camera_resolution[1] - crop_y, height + 2 * padding_y)

			# Adjust crop to maintain display aspect ratio (for rotated display)
			display_aspect = self.display_width / self.display_height  # 1080/1920 for rotated
			crop_aspect = crop_w / crop_h

			if crop_aspect > display_aspect:
				# Crop is wider than display, adjust height
				new_crop_h = crop_w / display_aspect
				height_diff = new_crop_h - crop_h
				crop_y = max(0, crop_y - height_diff / 2)
				crop_h = min(self.camera_resolution[1] - crop_y, new_crop_h)
				# If we hit the edge, adjust width instead
				if crop_y + crop_h > self.camera_resolution[1]:
					crop_h = self.camera_resolution[1] - crop_y
					crop_w = crop_h * display_aspect
					width_diff = (min(self.camera_resolution[0] - crop_x, crop_w) - crop_w) / 2
					crop_x = max(0, crop_x + width_diff)
			else:
				# Crop is taller than display, adjust width
				new_crop_w = crop_h * display_aspect
				width_diff = new_crop_w - crop_w
				crop_x = max(0, crop_x - width_diff / 2)
				crop_w = min(self.camera_resolution[0] - crop_x, new_crop_w)
				# If we hit the edge, adjust height instead
				if crop_x + crop_w > self.camera_resolution[0]:
					crop_w = self.camera_resolution[0] - crop_x
					crop_h = crop_w / display_aspect
					height_diff = (min(self.camera_resolution[1] - crop_y, crop_h) - crop_h) / 2
					crop_y = max(0, crop_y + height_diff)

			return (int(crop_x), int(crop_y), int(crop_w), int(crop_h))

		except Exception as e:
			print(f"Error calculating crop region: {e}")
			return None

	def posterize_frame(self, frame: np.ndarray) -> np.ndarray:
		"""Convert frame to pure black and white (no grays) - Fixed color handling"""
		# Fix color space if needed
		if frame.shape[2] == 3:
			# Assume it's already RGB from camera
			gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		else:
			gray = frame
		_, posterized = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
		return cv2.cvtColor(posterized, cv2.COLOR_GRAY2RGB)

	def freeze_and_start_evolution(self):
		"""
		Freeze current frame and start genetic evolution

		Process:
		1. Extract crop region based on pose detection
		2. Posterize to black and white
		3. Resize to target resolution
		4. Initialize genetic algorithm
		5. Start evolution
		"""
		if not self.pose_detected or self.current_frame is None:
			print("No pose detected - cannot freeze frame")
			return False

		crop_region_coords = self.current_frame_crop_region

		if crop_region_coords is not None:
			try:
				crop_x, crop_y, crop_w, crop_h = crop_region_coords
				crop_region = self.current_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

				if crop_region.size > 0:
					# Store frozen frame and posterize
					self.frozen_frame = crop_region.copy()
					self.frozen_posterized = self.posterize_frame(crop_region)

					# Store frozen keypoints for debug overlay
					if self.last_keypoints is not None and len(self.last_keypoints) > 0:
						best_person_idx = 0 if len(self.last_scores) == 1 else np.argmax(self.last_scores)
						if best_person_idx < len(self.last_keypoints):
							self.frozen_keypoints = self.last_keypoints[best_person_idx].copy()

					# Resize to target crop size for genetic algorithm
					target_image = cv2.resize(self.frozen_posterized, (self.crop_width, self.crop_height))

					# Get pose keypoints for detail mask
					pose_keypoints = None
					if self.last_scores is not None and self.last_keypoints is not None:
						if isinstance(self.last_scores, (list, np.ndarray)) and len(self.last_scores) > 0:
							best_person_idx = 0 if len(self.last_scores) == 1 else np.argmax(self.last_scores)
							if best_person_idx < len(self.last_keypoints):
								pose_keypoints = self.last_keypoints[best_person_idx]

					# Initialize genetic algorithm
					self.genetic_generator.initialize(target_image, pose_keypoints)
					self.evolution_active = True
					self.evolution_paused = False

					crop_mode_name = self.crop_modes[self.current_crop_mode]
					print(f"Frame frozen! Mode: {crop_mode_name}, Starting genetic evolution...")
					return True
			except Exception as e:
				print(f"Error freezing frame: {e}")

		print("Failed to freeze frame")
		return False

	def handle_events(self):
		"""Handle keyboard input events"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					return False
				elif event.key == pygame.K_SPACE:
					# Freeze frame and start evolution
					self.freeze_and_start_evolution()
				elif event.key == pygame.K_p:
					# Pause/resume evolution
					if self.evolution_active:
						self.evolution_paused = not self.evolution_paused
						status = "paused" if self.evolution_paused else "resumed"
						print(f"Evolution {status}")
				elif event.key == pygame.K_r:
					# Reset evolution - go back to live camera feed
					self.evolution_active = False
					self.evolution_paused = False
					self.frozen_frame = None
					self.frozen_posterized = None
					self.frozen_keypoints = None
					self.genetic_generator.mosaic = None
					self.genetic_generator.target_image = None
					self.genetic_generator.detail_mask = None
					self.genetic_generator.current_iteration = 0
					print("Evolution reset - back to live camera feed")
				elif event.key == pygame.K_d:
					# Cycle debug overlay modes
					self.current_debug_mode = (self.current_debug_mode + 1) % len(self.debug_modes)
					mode_name = self.debug_modes[self.current_debug_mode]
					print(f"Debug mode: {mode_name}")
				elif event.key == pygame.K_m:
					# Cycle crop mode
					old_mode = self.crop_modes[self.current_crop_mode]
					self.current_crop_mode = (self.current_crop_mode + 1) % len(self.crop_modes)
					new_mode = self.crop_modes[self.current_crop_mode]
					print(f"Crop mode: {old_mode} -> {new_mode}")
				elif event.key == pygame.K_s:
					# Save current mosaic
					if self.genetic_generator.mosaic is not None:
						filename = f"genetic_mosaic_iter_{self.genetic_generator.current_iteration}.png"
						mosaic_rgb = cv2.cvtColor(self.genetic_generator.mosaic, cv2.COLOR_GRAY2RGB)
						cv2.imwrite(filename, cv2.cvtColor(mosaic_rgb, cv2.COLOR_RGB2BGR))
						print(f"Saved {filename}")
		return True

	def draw_debug_overlay(self):
		"""
		Draw debug overlays based on current debug mode

		Creates semi-transparent overlay showing:
		- POSE: skeleton and keypoints
		- POSTERIZED: black & white target
		- MASK: detail mask for tile sizing
		"""
		debug_mode = self.debug_modes[self.current_debug_mode]

		if debug_mode == "OFF":
			return

		# Create semi-transparent overlay surface
		overlay = pygame.Surface((self.display_width, self.display_height), pygame.SRCALPHA)
		overlay.fill((0, 0, 0, 100))  # Semi-transparent black background

		if debug_mode == "POSE":
			self.draw_pose_overlay(overlay)
		elif debug_mode == "POSTERIZED":
			self.draw_posterized_overlay(overlay)
		elif debug_mode == "MASK":
			self.draw_mask_overlay(overlay)

		# Blit overlay to main screen
		self.screen.blit(overlay, (0, 0))

		# Draw debug mode indicator
		mode_text = self.font_medium.render(f"Debug: {debug_mode}", True, self.yellow)
		self.screen.blit(mode_text, (20, 20))

	def draw_pose_overlay(self, overlay_surface):
		"""Draw pose keypoints and skeleton on overlay - Uses frozen keypoints when evolution active"""
		# Use frozen keypoints if evolution is active, otherwise use live keypoints
		if self.evolution_active and self.frozen_keypoints is not None:
			person_keypoints = self.frozen_keypoints
			crop_region = self.current_frame_crop_region  # Use the crop from when frozen
		elif not self.pose_detected or self.last_keypoints is None:
			no_pose_text = self.font_large.render("NO POSE DETECTED", True, self.red)
			if self.rotate_90_degrees:
				no_pose_text = pygame.transform.rotate(no_pose_text, 90)
			text_rect = no_pose_text.get_rect(center=(self.display_width//2, self.display_height//2))
			overlay_surface.blit(no_pose_text, text_rect)
			return
		else:
			# Get best person from live keypoints
			best_person_idx = 0
			if isinstance(self.last_scores, (list, np.ndarray)) and len(self.last_scores) > 1:
				best_person_idx = np.argmax(self.last_scores)

			if best_person_idx >= len(self.last_keypoints):
				return

			person_keypoints = self.last_keypoints[best_person_idx]
			crop_region = self.current_frame_crop_region

		try:
			# COCO skeleton connections
			skeleton = [
				(0, 1), (0, 2),  # nose to eyes
				(1, 3), (2, 4),  # eyes to ears
				(0, 5), (0, 6),  # nose to shoulders
				(5, 6),          # shoulder to shoulder
				(5, 7), (6, 8),  # shoulders to elbows
				(7, 9), (8, 10), # elbows to wrists
				(5, 11), (6, 12), # shoulders to hips
				(11, 12),        # hip to hip
				(11, 13), (12, 14), # hips to knees
				(13, 15), (14, 16)  # knees to ankles
			]

			# Convert keypoints to display coordinates (matching video transformation)
			screen_points = {}
			for idx, keypoint in enumerate(person_keypoints):
				if len(keypoint) >= 3:
					kp_x, kp_y, visibility = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])

					if visibility > self.visibility_threshold:
						# Transform coordinates from camera space to final display space
						if crop_region:
							crop_x, crop_y, crop_w, crop_h = crop_region
							# Convert to cropped coordinates first
							relative_x = (kp_x - crop_x) / crop_w
							relative_y = (kp_y - crop_y) / crop_h

							# Scale to content area size (before rotation)
							target_x = relative_x * self.crop_width
							target_y = relative_y * self.crop_height
						else:
							# No crop, scale directly from camera to content area
							target_x = (kp_x / self.camera_resolution[0]) * self.crop_width
							target_y = (kp_y / self.camera_resolution[1]) * self.crop_height

						# Apply rotation if needed (same as video)
						if self.rotate_90_degrees:
							# 90° rotation: (x,y) -> (y, width-x)
							rotated_x = target_y
							rotated_y = self.crop_width - target_x
							final_x = rotated_x
							final_y = rotated_y
						else:
							final_x = target_x
							final_y = target_y

						# Add margin offsets to place in display area
						screen_x = self.margin_left + final_x
						screen_y = self.margin_top + final_y

						# Clamp to display bounds
						screen_x = max(self.margin_left, min(self.margin_left + self.crop_width - 1, screen_x))
						screen_y = max(self.margin_top, min(self.margin_top + self.crop_height - 1, screen_y))

						screen_points[idx] = (int(screen_x), int(screen_y))

			# Draw skeleton lines
			line_width = 3
			for start_idx, end_idx in skeleton:
				if start_idx in screen_points and end_idx in screen_points:
					pygame.draw.line(overlay_surface, self.white,
								   screen_points[start_idx], screen_points[end_idx], line_width)

			# Draw keypoints with labels
			keypoint_names = [
				'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
				'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
				'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
				'L_knee', 'R_knee', 'L_ankle', 'R_ankle'
			]

			keypoint_radius = 8
			for idx in screen_points:
				# Color code keypoints by body part
				if idx == 0:  # nose
					color = self.red
				elif idx in [1, 2, 3, 4]:  # face
					color = self.yellow
				elif idx in [5, 6]:  # shoulders
					color = self.green
				else:  # body
					color = self.white

				pygame.draw.circle(overlay_surface, color, screen_points[idx], keypoint_radius)

				# Draw label with rotation for readability
				if idx < len(keypoint_names):
					rotation_angle = 90 if self.rotate_90_degrees else 0
					label = self.draw_rotated_text(keypoint_names[idx], self.font_small, color, rotation_angle)
					label_x = screen_points[idx][0] + 12
					label_y = screen_points[idx][1] - 12
					overlay_surface.blit(label, (label_x, label_y))

			# Status indicator for frozen vs live
			status_text = "FROZEN POSE" if (self.evolution_active and self.frozen_keypoints is not None) else "LIVE POSE"
			status_color = self.yellow if self.evolution_active else self.green
			rotation_angle = 90 if self.rotate_90_degrees else 0
			status_surface = self.draw_rotated_text(status_text, self.font_medium, status_color, rotation_angle)
			overlay_surface.blit(status_surface, (self.margin_left + 20, self.margin_top + 60))

		except Exception as e:
			error_text = self.font_medium.render(f"Pose overlay error: {str(e)[:50]}", True, self.red)
			overlay_surface.blit(error_text, (20, 100))

	def draw_posterized_overlay(self, overlay_surface):
		"""Draw posterized target image on overlay - Fixed to match video rotation and framing"""
		if self.evolution_active and self.genetic_generator.target_image is not None:
			# Show frozen posterized target (properly sized)
			target_img = self.frozen_posterized  # Use original frozen image, not resized
		elif self.current_frame is not None and self.current_frame_crop_region is not None:
			# Show live posterized crop
			crop_x, crop_y, crop_w, crop_h = self.current_frame_crop_region
			crop_region = self.current_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
			target_img = self.posterize_frame(crop_region)
		else:
			no_target_text = self.font_large.render("NO TARGET IMAGE", True, self.red)
			if self.rotate_90_degrees:
				no_target_text = pygame.transform.rotate(no_target_text, 90)
			text_rect = no_target_text.get_rect(center=(self.display_width//2, self.display_height//2))
			overlay_surface.blit(no_target_text, text_rect)
			return

		try:
			# Convert to pygame surface
			if len(target_img.shape) == 3:
				target_surface = pygame.surfarray.make_surface(np.transpose(target_img, (1, 0, 2)))
			else:
				target_surface = pygame.surfarray.make_surface(target_img.T)

			# Scale to fit main content area with aspect ratio
			scaled_target = self.scale_with_aspect_ratio(target_surface, self.crop_width, self.crop_height)

			# Apply same rotation as video
			if self.rotate_90_degrees:
				scaled_target = pygame.transform.rotate(scaled_target, 90)

			overlay_surface.blit(scaled_target, (self.margin_left, self.margin_top))

			# Draw label with rotation
			rotation_angle = 90 if self.rotate_90_degrees else 0
			label_text = self.draw_rotated_text("POSTERIZED TARGET", self.font_medium, self.white, rotation_angle)
			overlay_surface.blit(label_text, (self.margin_left + 20, self.margin_top + 20))

		except Exception as e:
			error_text = self.font_medium.render(f"Posterized overlay error: {str(e)[:50]}", True, self.red)
			overlay_surface.blit(error_text, (20, 100))

	def draw_mask_overlay(self, overlay_surface):
		"""Draw detail mask showing tile size distribution - Fixed to match video rotation and framing"""
		if self.genetic_generator.detail_mask is None:
			no_mask_text = self.font_large.render("NO DETAIL MASK", True, self.red)
			if self.rotate_90_degrees:
				no_mask_text = pygame.transform.rotate(no_mask_text, 90)
			text_rect = no_mask_text.get_rect(center=(self.display_width//2, self.display_height//2))
			overlay_surface.blit(no_mask_text, text_rect)
			return

		try:
			# Convert grayscale mask to pygame surface properly
			mask = self.genetic_generator.detail_mask

			# Convert grayscale to RGB for pygame
			mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
			mask_surface = pygame.surfarray.make_surface(np.transpose(mask_rgb, (1, 0, 2)))

			# Scale to fit main content area with aspect ratio
			scaled_mask = self.scale_with_aspect_ratio(mask_surface, self.crop_width, self.crop_height)

			# Apply same rotation as video
			if self.rotate_90_degrees:
				scaled_mask = pygame.transform.rotate(scaled_mask, 90)

			overlay_surface.blit(scaled_mask, (self.margin_left, self.margin_top))

			# Draw label with rotation
			rotation_angle = 90 if self.rotate_90_degrees else 0
			label_text = self.draw_rotated_text("DETAIL MASK", self.font_medium, self.white, rotation_angle)
			overlay_surface.blit(label_text, (self.margin_left + 20, self.margin_top + 20))

			# Draw legend explaining mask colors with rotation
			legend_y = self.margin_top + 60
			legend_texts = [
				("White: 38px tiles (high detail)", self.white),
				("Gray: 76px tiles (medium detail)", (128, 128, 128)),
				("Black: 152px tiles (low detail)", (64, 64, 64))
			]

			for text, color in legend_texts:
				legend_surface = self.draw_rotated_text(text, self.font_small, color, rotation_angle)
				overlay_surface.blit(legend_surface, (self.margin_left + 20, legend_y))
				legend_y += 25

		except Exception as e:
			error_text = self.font_medium.render(f"Mask overlay error: {str(e)[:50]}", True, self.red)
			overlay_surface.blit(error_text, (20, 100))

	def draw_info_overlay(self):
		"""Draw information overlay in bottom margin - Added mutation progress"""
		info_y = self.display_height - self.margin_bottom + 5

		# Evolution status
		if self.evolution_active:
			if self.evolution_paused:
				status = "PAUSED"
				status_color = self.yellow
			else:
				status = "EVOLVING"
				status_color = self.green
		else:
			status = "WAITING"
			status_color = self.red

		status_text = self.font_small.render(f"Status: {status}", True, status_color)
		self.screen.blit(status_text, (self.margin_left, info_y))

		# Evolution statistics with mutation progress
		if self.genetic_generator.mosaic is not None:
			iter_text = self.font_small.render(f"Iter: {self.genetic_generator.current_iteration}", True, self.white)
			self.screen.blit(iter_text, (self.margin_left + 120, info_y))

			# Calculate mutation progress percentage
			max_iter = self.genetic_generator.max_iterations
			current_iter = self.genetic_generator.current_iteration
			progress_percent = min(100, (current_iter / max_iter) * 100) if max_iter > 0 else 0

			progress_text = self.font_small.render(f"Progress: {progress_percent:.1f}%", True, self.white)
			self.screen.blit(progress_text, (self.margin_left + 220, info_y))

			score_text = self.font_small.render(f"MSE: {self.genetic_generator.best_score:.1f}", True, self.white)
			self.screen.blit(score_text, (self.margin_left + 340, info_y))

			regions_text = self.font_small.render(f"Regions: {len(self.genetic_generator.regions)}", True, self.white)
			self.screen.blit(regions_text, (self.margin_left + 440, info_y))

		# Pose and crop info
		pose_status = "POSE" if self.pose_detected else "NO POSE"
		pose_color = self.green if self.pose_detected else self.red
		pose_text = self.font_small.render(pose_status, True, pose_color)
		self.screen.blit(pose_text, (self.margin_left + 550, info_y))

		crop_mode = self.crop_modes[self.current_crop_mode]
		crop_text = self.font_small.render(f"Crop: {crop_mode}", True, self.white)
		self.screen.blit(crop_text, (self.margin_left + 620, info_y))

		# FPS display
		fps_text = self.font_small.render(f"FPS: {self.current_fps:.1f}", True, self.white)
		self.screen.blit(fps_text, (self.margin_left + 750, info_y))

		# Camera resolution info
		cam_res_text = self.font_small.render(f"Cam: {self.camera_resolution[0]}x{self.camera_resolution[1]}", True, self.white)
		self.screen.blit(cam_res_text, (self.margin_left + 820, info_y))

	def draw_rotated_text(self, text, font, color, angle=0):
		"""Draw text with rotation for better readability on rotated TV"""
		text_surface = font.render(text, True, color)
		if angle != 0:
			text_surface = pygame.transform.rotate(text_surface, angle)
		return text_surface

	def scale_with_aspect_ratio(self, surface, target_width, target_height):
		"""Scale surface to fit target dimensions while maintaining aspect ratio"""
		source_width, source_height = surface.get_size()
		source_aspect = source_width / source_height
		target_aspect = target_width / target_height

		if source_aspect > target_aspect:
			# Source is wider, fit to width
			new_width = target_width
			new_height = int(target_width / source_aspect)
		else:
			# Source is taller, fit to height
			new_height = target_height
			new_width = int(target_height * source_aspect)

		# Scale the surface
		scaled_surface = pygame.transform.scale(surface, (new_width, new_height))

		# Create final surface with black background
		final_surface = pygame.Surface((target_width, target_height))
		final_surface.fill(self.black)

		# Center the scaled surface
		x_offset = (target_width - new_width) // 2
		y_offset = (target_height - new_height) // 2
		final_surface.blit(scaled_surface, (x_offset, y_offset))

		return final_surface

	def draw_frame(self):
		"""
		Main drawing function - Fixed coordinate transformations and rotation

		Displays either:
		- Live camera crop (when waiting)
		- Evolving mosaic (when evolution active)
		Plus debug overlays and info text
		"""
		self.update_fps()
		self.screen.fill(self.black)

		# Main display area (with exact margins)
		main_rect = pygame.Rect(self.margin_left, self.margin_top, self.crop_width, self.crop_height)

		if self.evolution_active and self.genetic_generator.mosaic is not None:
			# Show evolving genetic mosaic
			mosaic_surface = pygame.surfarray.make_surface(self.genetic_generator.mosaic.T)

			# Apply rotation if needed (to match TV orientation)
			if self.rotate_90_degrees:
				mosaic_surface = pygame.transform.rotate(mosaic_surface, 90)

			self.screen.blit(mosaic_surface, main_rect.topleft)

		elif self.current_frame is not None:
			# Show live camera with proper crop transformation
			if self.current_frame_crop_region is not None:
				crop_x, crop_y, crop_w, crop_h = self.current_frame_crop_region

				# Extract crop region from camera frame
				crop_region = self.current_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

				if crop_region.size > 0:
					# Fix color inversion: Convert BGR to RGB
					crop_region_rgb = cv2.cvtColor(crop_region, cv2.COLOR_BGR2RGB)

					# Convert to pygame surface (note the transpose for correct orientation)
					frame_surface = pygame.surfarray.make_surface(np.transpose(crop_region_rgb, (1, 0, 2)))

					# Scale to fit main area exactly
					scaled_frame = pygame.transform.scale(frame_surface, (self.crop_width, self.crop_height))

					# Apply rotation if needed (to match TV orientation)
					if self.rotate_90_degrees:
						scaled_frame = pygame.transform.rotate(scaled_frame, 90)

					self.screen.blit(scaled_frame, main_rect.topleft)
				else:
					# Show full frame if crop fails
					current_frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
					frame_surface = pygame.surfarray.make_surface(np.transpose(current_frame_rgb, (1, 0, 2)))
					scaled_frame = pygame.transform.scale(frame_surface, (self.crop_width, self.crop_height))

					if self.rotate_90_degrees:
						scaled_frame = pygame.transform.rotate(scaled_frame, 90)

					self.screen.blit(scaled_frame, main_rect.topleft)
			else:
				# Show full frame when no crop region
				current_frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
				frame_surface = pygame.surfarray.make_surface(np.transpose(current_frame_rgb, (1, 0, 2)))
				scaled_frame = pygame.transform.scale(frame_surface, (self.crop_width, self.crop_height))

				if self.rotate_90_degrees:
					scaled_frame = pygame.transform.rotate(scaled_frame, 90)

				self.screen.blit(scaled_frame, main_rect.topleft)
		else:
			# Show loading message
			loading_text = self.font_large.render("Initializing camera...", True, self.white)
			text_rect = loading_text.get_rect(center=main_rect.center)
			self.screen.blit(loading_text, text_rect)

		# Draw debug overlay if active
		self.draw_debug_overlay()

	def run(self):
		"""
		Main application loop

		Process:
		1. Handle keyboard events
		2. Run genetic evolution steps (if active)
		3. Draw frame and overlays
		4. Update display at 30 FPS
		"""
		running = True

		while running:
			running = self.handle_events()

			# Run genetic evolution steps
			if self.evolution_active and not self.evolution_paused:
				for _ in range(self.steps_per_frame):
					improved = self.genetic_generator.evolve_step()
					# Print progress every 500 iterations
					if improved and self.genetic_generator.current_iteration % 500 == 0:
						print(f"Iteration {self.genetic_generator.current_iteration}, MSE: {self.genetic_generator.best_score:.2f}")

			self.draw_frame()
			pygame.display.flip()
			self.clock.tick(30)  # 30 FPS

		# Cleanup
		if self.picam2:
			self.picam2.stop()
		pygame.quit()

if __name__ == "__main__":
	try:
		app = GeneticTileArtApp()
		app.run()
	except KeyboardInterrupt:
		print("\nExiting...")
	except Exception as e:
		print(f"Error: {e}")
		print("Make sure you're running on a Raspberry Pi with IMX500 camera")
		print("Place your geometric tile images (*.png) in the 'tiles' directory")