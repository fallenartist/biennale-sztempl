# python/sztempl/capture.py

import sys
import time
import logging
import os

import numpy as np
import pygame
import cv2
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

from sztempl.config import settings
from sztempl.tile_filter import TileFilter

logger = logging.getLogger(__name__)

class FaceShoulderApp:
	def __init__(self):
		# ==== CONFIGURABLE (can be overridden via config["capture"]) ====
		cfg = settings.get("capture") or {}

		# Full HD dimensions for rotation logic
		self.fullhd_width  = cfg.get("fullhd_width", 1920)
		self.fullhd_height = cfg.get("fullhd_height", 1080)
		# Default portrait orientation (rotate 90°)
		self.rotate_90_degrees    = cfg.get("rotation", True)
		# Mirror before rotation
		self.mirror_horizontally  = cfg.get("mirror", True)

		# Compute display dims based on rotation
		if self.rotate_90_degrees:
			self.display_width  = self.fullhd_height
			self.display_height = self.fullhd_width
		else:
			self.display_width  = self.fullhd_width
			self.display_height = self.fullhd_height

		# Pose thresholds
		self.visibility_threshold = 0.3
		self.detection_threshold  = 0.3
		self.crop_padding_factor  = 0.2

		# Camera settings
		self.camera_resolution    = tuple(cfg.get("resolution", (2028, 1520)))
		self.pose_model_path      = cfg.get(
			"pose_model_path",
			"/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
		)
		# Window size (height, width) for pygame
		self.WINDOW_SIZE_H_W      = (self.camera_resolution[1], self.camera_resolution[0])
		self.inference_fps        = cfg.get("framerate", 30)

		# FPS tracking
		self.frame_times = []
		self.frame_count = 0
		self.current_fps = 0.0

		# Colors
		self.black = (0, 0, 0)
		self.white = (255, 255, 255)
		self.offwhite = (225, 225, 225)
		self.green = (0, 255, 0)
		self.red   = (255, 0, 0)

		# Pygame setup
		pygame.init()
		self.screen = pygame.display.set_mode(
			(self.display_width, self.display_height),
			pygame.FULLSCREEN
		)
		pygame.display.set_caption("Face & Shoulders Detection")

		# State
		self.last_keypoints = None
		self.last_scores    = None
		self.pose_detected  = False
		self.current_frame  = None
		self.debug_mode     = False

		# Pose detection stabilization - CONFIGURABLE VARIABLES
		self.pose_stability_window = 30   # Frames to consider for stability
		self.pose_switch_threshold = 0.7  # 70% of frames must agree to switch
		self.min_mode_duration = 3.0      # Minimum seconds before allowing mode switch

		# Stabilization state (don't change these)
		self.pose_detection_history = []  # Track recent detections
		self.stable_pose_detected = False  # The stable state we use for display
		self.last_mode_switch_time = 0     # When we last switched modes

		# Camera init
		self.setup_imx500_camera()

		self.clock = pygame.time.Clock()

		# Tiles - Using larger tile sizes with eye tile feature
		tiles_dir = os.path.join(os.path.dirname(__file__), "tiles")
		self.tile_filter = TileFilter(
			tiles_dir,
			cell_sizes=[40, 80, 120],
			enable_rotation=False,
			enable_noise_random_rotation=True,
			noise_rotation_interval=60,
			enable_filter_random_rotation=False,
			filter_rotation_interval=45,
			eye_tile=True
		)

		# Configure tile distributions
		self.tile_filter.set_noise_size_distribution({40: 1, 80: 4, 120: 3})
		self.tile_filter.set_filter_size_distribution({40: 6, 80: 3, 120: 1})

		# Genetic mutation module
		try:
			from sztempl.genetic_mutation import GeneticMutationModule
			self.genetic_module = GeneticMutationModule(tiles_dir, gpio_pin=18)
			self.genetic_enabled = True
		except ImportError as e:
			self.genetic_module = None
			self.genetic_enabled = False

		print(f"Display: {self.display_width}x{self.display_height}, "
			  f"rotate90={self.rotate_90_degrees}, mirror={self.mirror_horizontally}")
		print("ESC: exit | d: debug toggle")
		if self.genetic_enabled:
			print("SPACE: toggle genetic mutation mode")

	def setup_imx500_camera(self):
		try:
			# Initialize IMX500 with pose model
			self.imx500 = IMX500(self.pose_model_path)
			self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
			self.intrinsics.task = "pose estimation"
			self.intrinsics.inference_rate = self.intrinsics.inference_rate or self.inference_fps
			self.intrinsics.update_with_defaults()

			# Configure Picamera2
			self.picam2 = Picamera2(self.imx500.camera_num)
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"},
				controls={'FrameRate': self.intrinsics.inference_rate},
				buffer_count=12
			)
			self.picam2.pre_callback = self.ai_output_tensor_parse
			self.imx500.show_network_fw_progress_bar()
			# Disable GPU preview
			self.picam2.start(config, show_preview=False)
			self.imx500.set_auto_aspect_ratio()

			# Capture the actual resolution
			self.actual_camera_resolution = config['main']['size']
			self.WINDOW_SIZE_H_W = (
				self.actual_camera_resolution[1],
				self.actual_camera_resolution[0]
			)
			print("IMX500 ready:", self.actual_camera_resolution,
				  self.intrinsics.inference_rate)
		except Exception as e:
			print(f"Camera init fail: {e}", file=sys.stderr)
			# Fallback to plain Picamera2
			self.picam2 = Picamera2()
			cfg = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"}
			)
			self.picam2.configure(cfg)
			self.picam2.start()
			self.imx500 = None
			self.actual_camera_resolution = self.camera_resolution
			self.WINDOW_SIZE_H_W = (
				self.camera_resolution[1],
				self.camera_resolution[0]
			)

	def update_fps(self):
		now = time.time()
		self.frame_times.append(now)
		self.frame_count += 1
		if len(self.frame_times) > 30:
			self.frame_times.pop(0)
		if self.frame_count % 10 == 0 and len(self.frame_times) > 1:
			span = self.frame_times[-1] - self.frame_times[0]
			self.current_fps = (len(self.frame_times) - 1) / span if span > 0 else 0.0

	def ai_output_tensor_parse(self, request: CompletedRequest):
		try:
			with MappedArray(request, 'main') as m:
				self.current_frame = m.array.copy()
		except:
			pass

		if not self.imx500:
			return

		try:
			outputs = self.imx500.get_outputs(
				metadata=request.get_metadata(), add_batch=True
			)
			if outputs is not None:
				kps, scores, boxes = postprocess_higherhrnet(
					outputs=outputs,
					img_size=self.WINDOW_SIZE_H_W,
					img_w_pad=(0, 0), img_h_pad=(0, 0),
					detection_threshold=self.detection_threshold,
					network_postprocess=True
				)
				if scores is not None and len(scores) > 0:
					self.last_keypoints = (
						np.reshape(np.stack(kps, axis=0), (len(scores), 17, 3))
					)
					self.last_scores = scores
					self.pose_detected = True
					self.current_frame_crop_region = (
						self._calculate_face_shoulder_crop_region()
					)
				else:
					self.pose_detected = False
					self.current_frame_crop_region = None
			else:
				self.pose_detected = False
				self.current_frame_crop_region = None
		except Exception as e:
			print(f"Parse error: {e}", file=sys.stderr)
			self.pose_detected = False
			self.current_frame_crop_region = None

	def _calculate_face_shoulder_crop_region(self):
		if not self.pose_detected:
			return None
		idx = np.argmax(self.last_scores)
		pts = self.last_keypoints[idx]
		inds = [0, 1, 2, 3, 4, 5, 6]
		valid = [(x, y) for (x, y, v) in pts[inds] if v > self.visibility_threshold]
		if len(valid) < 2:
			return None
		xs, ys = zip(*valid)
		minx, maxx = min(xs), max(xs)
		miny, maxy = min(ys), max(ys)
		w, h = maxx - minx, maxy - miny
		padx, pady = w * self.crop_padding_factor, h * self.crop_padding_factor
		cx = max(0, minx - padx)
		cy = max(0, miny - pady)
		cw = min(self.camera_resolution[0] - cx, w + 2 * padx)
		ch = min(self.camera_resolution[1] - cy, h + 2 * pady)

		disp_aspect = self.display_width / self.display_height
		crop_aspect = cw / ch
		if crop_aspect > disp_aspect:
			new_h = cw / disp_aspect
			dy = (new_h - ch) / 2
			cy = max(0, cy - dy)
			ch = min(self.camera_resolution[1] - cy, new_h)
		else:
			new_w = ch * disp_aspect
			dx = (new_w - cw) / 2
			cx = max(0, cx - dx)
			cw = min(self.camera_resolution[0] - cx, new_w)

		return (int(cx), int(cy), int(cw), int(ch))

	def get_face_shoulder_crop_region(self):
		return getattr(self, 'current_frame_crop_region', None)

	def scale_with_aspect_ratio(self, surf, tw, th):
		sw, sh = surf.get_size()
		if sw / sh > tw / th:
			nw, nh = tw, int(tw * sh / sw)
		else:
			nh, nw = th, int(th * sw / sh)
		s = pygame.transform.scale(surf, (nw, nh))
		f = pygame.Surface((tw, th))
		f.fill(self.black)
		f.blit(s, ((tw - nw) // 2, (th - nh) // 2))
		return f

	def update_pose_stability(self):
		"""Update the stable pose detection state based on recent history"""
		import time

		# Add current detection to history
		self.pose_detection_history.append(self.pose_detected)

		# Keep only recent history
		if len(self.pose_detection_history) > self.pose_stability_window:
			self.pose_detection_history.pop(0)

		# Calculate percentage of recent frames with pose detection
		if len(self.pose_detection_history) >= self.pose_stability_window:
			detection_ratio = sum(self.pose_detection_history) / len(self.pose_detection_history)

			# Determine what the stable state should be
			should_be_detected = detection_ratio >= self.pose_switch_threshold
			should_be_noise = detection_ratio <= (1.0 - self.pose_switch_threshold)

			# Check if enough time has passed since last switch
			current_time = time.time()
			time_since_switch = current_time - self.last_mode_switch_time

			# Only switch if enough time has passed AND we have a clear decision
			if time_since_switch >= self.min_mode_duration:
				if should_be_detected and not self.stable_pose_detected:
					# Switch to filter mode
					self.stable_pose_detected = True
					self.last_mode_switch_time = current_time
					if self.debug_mode:
						print(f"Switched to FILTER mode (detection ratio: {detection_ratio:.2f})")

				elif should_be_noise and self.stable_pose_detected:
					# Switch to noise mode
					self.stable_pose_detected = False
					self.last_mode_switch_time = current_time
					if self.debug_mode:
						print(f"Switched to NOISE mode (detection ratio: {detection_ratio:.2f})")

	def get_processed_frame_for_tiles(self, use_person_crop=False):
		"""
		Process the frame for tile processing.

		use_person_crop: If True, crops around detected person (for genetic mutation).
						If False, uses center crop to TV proportions (default for tile filter).
		"""
		if self.current_frame is None:
			return None

		try:
			# Start with BGR to RGB conversion
			rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

			# Create pygame surface for processing
			surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

			# Apply framing logic based on use_person_crop parameter
			if use_person_crop:
				# OLD BEHAVIOR: Crop around detected person (for genetic mutation)
				crop = self.get_face_shoulder_crop_region()
				if crop and self.pose_detected:
					x, y, w, h = crop
					if w > 0 and h > 0:
						# Crop to face/shoulder region
						cs = pygame.Surface((w, h))
						cs.blit(surf, (0, 0), (x, y, w, h))
						# Scale to display size
						ds = pygame.transform.scale(
							cs, (self.display_width, self.display_height)
						)
					else:
						# Scale with aspect ratio
						ds = self.scale_with_aspect_ratio(
							surf, self.display_width, self.display_height
						)
				else:
					# Scale with aspect ratio
					ds = self.scale_with_aspect_ratio(
						surf, self.display_width, self.display_height
					)
			else:
				# NEW DEFAULT BEHAVIOR: Center crop to TV proportions (1080x1920)
				camera_w, camera_h = surf.get_size()

				# Calculate target crop size (TV proportions before rotation)
				target_aspect = self.display_height / self.display_width  # 1920/1080 = 16:9 portrait

				# Determine crop size based on camera aspect ratio
				if camera_w / camera_h > 1 / target_aspect:
					# Camera is wider - crop width to match height
					crop_h = camera_h
					crop_w = int(camera_h / target_aspect)
				else:
					# Camera is taller - crop height to match width
					crop_w = camera_w
					crop_h = int(camera_w * target_aspect)

				# Center the crop
				crop_x = (camera_w - crop_w) // 2
				crop_y = (camera_h - crop_h) // 2

				# Extract center crop
				cs = pygame.Surface((crop_w, crop_h))
				cs.blit(surf, (0, 0), (crop_x, crop_y, crop_w, crop_h))

				# Scale to display size (before rotation)
				ds = pygame.transform.scale(cs, (self.display_width, self.display_height))

			# Apply mirror and rotation transforms
			if self.mirror_horizontally:
				ds = pygame.transform.flip(ds, True, False)
			if self.rotate_90_degrees:
				ds = pygame.transform.rotate(ds, 90)

			# Convert back to numpy array for tile processing
			# pygame surface to numpy array
			frame_array = pygame.surfarray.array3d(ds)
			# Swap axes back to normal image format (height, width, channels)
			frame_array = frame_array.swapaxes(0, 1)

			return frame_array

		except Exception as e:
			print(f"Frame processing error: {e}", file=sys.stderr)
			return None

	def draw_video_display(self):
		"""
		Main drawing function with stabilized mode switching and genetic mutation
		"""
		self.update_fps()

		# Update pose stability (this determines the stable mode)
		self.update_pose_stability()

		# Clear screen
		self.screen.fill(self.offwhite)

		if self.current_frame is None:
			font = pygame.font.Font(None, 74)
			t = font.render("Initializing camera...", True, self.white)
			r = t.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(t, r)
			return

		# Check GPIO trigger regardless of genetic mode state
		if self.genetic_enabled and self.genetic_module and self.genetic_module.gpio_enabled:
			if self.genetic_module.check_gpio_trigger():
				self._toggle_genetic_mode()

		# Update genetic mutation if active
		if self.genetic_enabled and self.genetic_module and self.genetic_module.is_active:
			self.genetic_module.update()

			# Draw genetic mosaic instead of normal tiles
			display_rect = pygame.Rect(0, 0, self.display_width, self.display_height)
			self.genetic_module.draw_mosaic(self.screen, display_rect)

			# Draw debug mask in debug mode
			if self.debug_mode:
				self.genetic_module.draw_debug_mask(self.screen, display_rect)

			# Draw progress bar
			self.genetic_module.draw_progress_bar(self.screen)

		else:
			# Normal tile filter operation
			if not self.stable_pose_detected:
				# No person detected (stable) - show tile noise
				self.tile_filter.apply_noise(self.screen)
			else:
				# Person detected (stable) - get processed frame and apply tile mosaic
				processed_frame = self.get_processed_frame_for_tiles()
				if processed_frame is not None:
					# Pass face keypoints to tile filter for eye tile positioning
					# Transform keypoints through same center crop pipeline
					if self.pose_detected and self.last_keypoints is not None and len(self.last_keypoints) > 0:
						best_person_idx = np.argmax(self.last_scores)
						face_keypoints = self.last_keypoints[best_person_idx]

						# Pass center crop parameters instead of person crop
						camera_w, camera_h = self.camera_resolution
						target_aspect = self.display_height / self.display_width

						if camera_w / camera_h > 1 / target_aspect:
							crop_h = camera_h
							crop_w = int(camera_h / target_aspect)
						else:
							crop_w = camera_w
							crop_h = int(camera_w * target_aspect)

						crop_x = (camera_w - crop_w) // 2
						crop_y = (camera_h - crop_h) // 2
						center_crop_region = (crop_x, crop_y, crop_w, crop_h)

						self.tile_filter.set_face_keypoints(
							face_keypoints,
							camera_resolution=self.camera_resolution,
							crop_region=center_crop_region,  # Use center crop instead of person crop
							display_size=(self.display_width, self.display_height),
							mirror=self.mirror_horizontally,
							rotate_90=self.rotate_90_degrees
						)

					self.tile_filter.apply_tiles(self.screen, processed_frame)
				else:
					# Fallback to noise if processing fails
					self.tile_filter.apply_noise(self.screen)

		# Draw overlays
		if self.debug_mode:
			self.draw_pose_debug_overlay()
			self.draw_status_overlay()

	def draw_pose_debug_overlay(self):
		if not self.pose_detected:
			return
		idx = np.argmax(self.last_scores)
		pts = self.last_keypoints[idx]
		inds = [0, 1, 2, 3, 4, 5, 6]

		# For debug overlay, we need to transform keypoints through the same pipeline as the image
		camera_w, camera_h = self.camera_resolution

		# Calculate center crop parameters (same as in get_processed_frame_for_tiles)
		target_aspect = self.display_height / self.display_width  # 1920/1080 = portrait aspect

		if camera_w / camera_h > 1 / target_aspect:
			# Camera is wider - crop width to match height
			crop_h = camera_h
			crop_w = int(camera_h / target_aspect)
		else:
			# Camera is taller - crop height to match width
			crop_w = camera_w
			crop_h = int(camera_w * target_aspect)

		# Center the crop
		crop_x = (camera_w - crop_w) // 2
		crop_y = (camera_h - crop_h) // 2

		scr_pts = {}
		for j in inds:
			x, y, v = pts[j]
			if v > self.visibility_threshold:
				# Transform keypoint through same pipeline as image:
				# 1. Apply center crop
				x_cropped = x - crop_x
				y_cropped = y - crop_y

				# Skip if outside crop region
				if x_cropped < 0 or x_cropped >= crop_w or y_cropped < 0 or y_cropped >= crop_h:
					continue

				# 2. Scale to display size (before rotation)
				sx = x_cropped * self.display_width / crop_w
				sy = y_cropped * self.display_height / crop_h

				# 3. Apply mirror transform
				if self.mirror_horizontally:
					sx = self.display_width - sx

				# 4. Apply rotation transform
				if self.rotate_90_degrees:
					# 90 degree CCW rotation: (x,y) -> (-y, x) but adjust for screen coordinates
					new_x = sy
					new_y = self.display_width - sx
					sx, sy = new_x, new_y

				scr_pts[j] = (int(sx), int(sy))

		# Draw pose skeleton
		for a, b in [(0,1), (0,2), (1,3), (2,4)]:
			if a in scr_pts and b in scr_pts:
				pygame.draw.line(self.screen, self.white,
								 scr_pts[a], scr_pts[b], 2)
		if 5 in scr_pts and 6 in scr_pts:
			pygame.draw.line(self.screen, self.green,
							 scr_pts[5], scr_pts[6], 2)
		for j in scr_pts:
			col = self.red if j == 0 else (self.white if j < 5 else self.green)
			pygame.draw.circle(self.screen, col, scr_pts[j], 8)

	def draw_status_overlay(self):
		"""Status overlay with stability info in debug mode"""
		font = pygame.font.Font(None, 48)

		# FPS display
		fps_text = f"FPS: {self.current_fps:.1f}"
		if self.rotate_90_degrees:
			txt = pygame.transform.rotate(font.render(fps_text, True, self.white), 90)
			pos = (self.display_width - 20, 20)
		else:
			txt = font.render(fps_text, True, self.white)
			pos = (self.display_width - 200, 20)
		self.screen.blit(txt, pos)

		# Camera resolution
		if hasattr(self, 'actual_camera_resolution'):
			res = f"CAM: {self.actual_camera_resolution[0]}x{self.actual_camera_resolution[1]}"
			if self.rotate_90_degrees:
				rt = pygame.transform.rotate(font.render(res, True, self.white), 90)
				rp = (self.display_width - 20, 20 + txt.get_width() + 10)
			else:
				rt = font.render(res, True, self.white)
				rp = (self.display_width - 300, 70)
			self.screen.blit(rt, rp)

		# Show genetic mode status
		if self.genetic_enabled and self.genetic_module:
			if self.genetic_module.is_active:
				mode_text = "GENETIC"
				mode_color = (255, 100, 0)  # Orange
			else:
				mode_text = f"{'FILTER' if self.stable_pose_detected else 'NOISE'}"
				mode_color = self.green if self.stable_pose_detected else self.white

			if self.rotate_90_degrees:
				mt = pygame.transform.rotate(font.render(mode_text, True, mode_color), 90)
				mp = (self.display_width - 20, 20 + txt.get_width() + rt.get_width() + 20)
			else:
				mt = font.render(mode_text, True, mode_color)
				mp = (self.display_width - 400, 120)
			self.screen.blit(mt, mp)

		# Show stability info only in debug mode
		if self.debug_mode:
			if len(self.pose_detection_history) >= self.pose_stability_window:
				detection_ratio = sum(self.pose_detection_history) / len(self.pose_detection_history)
				status_text = f"({detection_ratio:.0%})"
			else:
				status_text = "(stabilizing...)"

			# Color code the status
			status_color = self.green if self.stable_pose_detected else self.white

			if self.rotate_90_degrees:
				st = pygame.transform.rotate(font.render(status_text, True, status_color), 90)
				if self.genetic_enabled and self.genetic_module:
					sp = (self.display_width - 20, 20 + txt.get_width() + rt.get_width() + mt.get_width() + 30)
				else:
					sp = (self.display_width - 20, 20 + txt.get_width() + rt.get_width() + 20)
			else:
				st = font.render(status_text, True, status_color)
				sp = (self.display_width - 500, 170)
			self.screen.blit(st, sp)

	def handle_events(self):
		"""Event handling - debug toggle and genetic mutation space key detection"""
		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				return False
			if e.type == pygame.KEYDOWN:
				if e.key == pygame.K_ESCAPE:
					return False
				if e.key == pygame.K_d:
					self.debug_mode = not self.debug_mode
					print("Debug mode", self.debug_mode)

				# Handle SPACE key for genetic mutation toggle
				if e.key == pygame.K_SPACE and self.genetic_enabled and self.genetic_module:
					self._toggle_genetic_mode()
		return True

	def _toggle_genetic_mode(self):
		"""Handle genetic mode toggle"""
		# Get processed frame for genetic mode with person cropping enabled
		processed_frame = self.get_processed_frame_for_tiles(use_person_crop=True)

		# Toggle genetic mode - use display size for both camera and target since frame is pre-processed
		target_size = (self.display_width, self.display_height)

		self.genetic_module.toggle_mode(
			processed_frame,
			None,  # No keypoints needed for pure Sobel approach
			target_size,  # Use target_size as camera_resolution since frame is already processed
			target_size
		)

	def run(self):
		# Main blocking loop, exactly like original
		while True:
			if not self.handle_events():
				break
			self.draw_video_display()
			pygame.display.flip()
			self.clock.tick(self.inference_fps)
		try:
			self.picam2.stop()
		except:
			pass
		pygame.quit()
		logger.info("Camera loop exited and Pygame quit")

def register(app_context):
	"""Plugin hook: instantiate and run on start()."""
	app = FaceShoulderApp()
	# We bypass threading to ensure all pygame calls stay on the main thread
	app_context.register_module(app)
	# Replace app.start() to call run() directly
	app.start = app.run
	logger.info("✅ FaceShoulderApp (capture) registered")