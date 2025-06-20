#!/usr/bin/env python3
"""
Face Analysis App - TV Landscape Mode - FIXED VERSION
- Video feed and pose estimation properly aligned
- Fixed rotation and cropping issues
- Consistent coordinate transformations
"""

import numpy as np
import pygame
import cv2
import time
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
import sys

class FaceAnalysisApp:
	def __init__(self):
		# ================================================
		# FIXED CONFIGURATION
		# ================================================

		# Display settings - ACTUAL TV SIZE (landscape)
		self.tv_width = 1920   # Actual TV width
		self.tv_height = 1080  # Actual TV height

		# The app renders to the actual TV size
		self.display_width = self.tv_width    # 1920
		self.display_height = self.tv_height  # 1080

		# Margins (in TV coordinates - landscape orientation)
		# 88px margin on RIGHT of unrotated TV (appears at bottom when rotated)
		self.margin_left = 8
		self.margin_top = 8
		self.margin_right = 88  # Changed from 50px to 88px
		self.margin_bottom = 8

		# Analysis area after margins: 1824 x 1064
		self.analysis_width = self.display_width - self.margin_left - self.margin_right   # 1824
		self.analysis_height = self.display_height - self.margin_top - self.margin_bottom  # 1064

		# Grid sizes for different zones
		self.face_grid_size = 38      # High detail for face area
		self.shoulder_grid_size = 76  # Medium detail for shoulder/upper body
		self.background_grid_size = 152  # Low detail for background

		# Pose detection thresholds
		self.visibility_threshold = 0.3
		self.detection_threshold = 0.3

		# Crop settings
		self.crop_padding_factor = 0.2

		# Camera settings
		self.camera_resolution = (2028, 1520)
		self.pose_model_path = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
		self.WINDOW_SIZE_H_W = (self.camera_resolution[1], self.camera_resolution[0])

		# Threshold settings for two-color processing
		self.threshold_value = 127  # Adjustable threshold (0-255)

		# Colors
		self.black = (0, 0, 0)
		self.white = (255, 255, 255)
		self.red = (255, 0, 0)
		self.green = (0, 255, 0)

		# FPS tracking
		self.frame_times = []
		self.fps_update_interval = 30
		self.frame_count = 0
		self.current_fps = 0.0

		# DEBUG: Orientation testing flags
		self.video_rotate_90 = True      # Whether to rotate video 90° (KEEP TRUE - correct orientation)
		self.video_flip_h = True         # Whether to flip video horizontally (KEEP TRUE - correct)
		self.crop_aspect_rotated = False # Whether crop uses rotated screen aspect ratio
		self.pose_rotate_90 = False      # Whether to rotate pose points 90°
		self.pose_flip_h = True          # Whether to flip pose points horizontally

		# ================================================
		# INITIALIZATION
		# ================================================

		pygame.init()
		# Render to actual TV size (landscape)
		self.screen = pygame.display.set_mode((self.display_width, self.display_height), pygame.FULLSCREEN)
		pygame.display.set_caption("Face Analysis - TV Landscape")

		# Pose detection variables
		self.last_keypoints = None
		self.last_scores = None
		self.last_boxes = None
		self.pose_detected = False
		self.current_frame = None
		self.current_frame_crop_region = None

		# Analysis grid data
		self.analysis_areas = []
		self.face_center = None

		self.setup_imx500_camera()
		self.clock = pygame.time.Clock()
		self.debug_mode = False

		print(f"TV Display: {self.display_width}x{self.display_height} (landscape)")
		print(f"Analysis area: {self.analysis_width}x{self.analysis_height}")
		print(f"Grid sizes: Face={self.face_grid_size}px, Shoulder={self.shoulder_grid_size}px, Background={self.background_grid_size}px")

	def setup_imx500_camera(self):
		"""Initialize IMX500 camera with pose estimation model"""
		try:
			self.imx500 = IMX500(self.pose_model_path)
			self.intrinsics = self.imx500.network_intrinsics

			if not self.intrinsics:
				self.intrinsics = NetworkIntrinsics()
				self.intrinsics.task = "pose estimation"
			elif self.intrinsics.task != "pose estimation":
				print("Warning: Network is not a pose estimation task", file=sys.stderr)

			if self.intrinsics.inference_rate is None:
				self.intrinsics.inference_rate = 30

			self.intrinsics.update_with_defaults()

			self.picam2 = Picamera2(self.imx500.camera_num)
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"},
				controls={'FrameRate': self.intrinsics.inference_rate},
				buffer_count=12
			)

			self.picam2.pre_callback = self.ai_output_tensor_parse
			self.imx500.show_network_fw_progress_bar()
			self.picam2.start(config, show_preview=False)
			self.imx500.set_auto_aspect_ratio()

			self.actual_camera_resolution = config['main']['size']
			self.WINDOW_SIZE_H_W = (self.actual_camera_resolution[1], self.actual_camera_resolution[0])

			print("IMX500 pose estimation initialized successfully")
			print(f"Camera resolution: {self.actual_camera_resolution}")

		except Exception as e:
			print(f"Failed to initialize IMX500: {e}")
			self.picam2 = Picamera2()
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"}
			)
			self.picam2.configure(config)
			self.picam2.start()
			self.imx500 = None
			self.actual_camera_resolution = self.camera_resolution
			self.WINDOW_SIZE_H_W = (self.actual_camera_resolution[1], self.actual_camera_resolution[0])

	def update_fps(self):
		"""Update FPS calculation"""
		current_time = time.time()
		self.frame_times.append(current_time)
		self.frame_count += 1

		if len(self.frame_times) > self.fps_update_interval:
			self.frame_times.pop(0)

		if self.frame_count % 10 == 0 and len(self.frame_times) > 1:
			time_span = self.frame_times[-1] - self.frame_times[0]
			if time_span > 0:
				self.current_fps = (len(self.frame_times) - 1) / time_span

	def ai_output_tensor_parse(self, request: CompletedRequest):
		"""Parse pose estimation output tensor and capture frame"""
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
					self.last_boxes = [np.array(b) for b in boxes]
					self.last_scores = scores
					self.pose_detected = True
					self.current_frame_crop_region = self._calculate_face_shoulder_crop_region()
					self._update_face_center()
				else:
					self.pose_detected = False
					self.current_frame_crop_region = None
					self.face_center = None
			else:
				self.pose_detected = False
				self.current_frame_crop_region = None
				self.face_center = None

		except Exception as e:
			print(f"Error parsing pose results: {e}")
			self.pose_detected = False
			self.current_frame_crop_region = None
			self.face_center = None

	def _update_face_center(self):
		"""Calculate face center in analysis coordinates - FIXED for analysis area scaling"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores) == 0:
			self.face_center = None
			return

		try:
			best_person_idx = np.argmax(self.last_scores)
			if best_person_idx >= len(self.last_keypoints):
				self.face_center = None
				return

			person_keypoints = self.last_keypoints[best_person_idx]
			if len(person_keypoints) < 17:
				self.face_center = None
				return

			# Face keypoint indices: 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
			face_indices = [0, 1, 2, 3, 4]
			valid_face_points = []

			for idx in face_indices:
				if idx < len(person_keypoints):
					kp_x, kp_y, visibility = person_keypoints[idx]
					if visibility > self.visibility_threshold:
						valid_face_points.append((kp_x, kp_y))

			if len(valid_face_points) >= 2:
				# Calculate face center in camera coordinates
				face_x = sum(p[0] for p in valid_face_points) / len(valid_face_points)
				face_y = sum(p[1] for p in valid_face_points) / len(valid_face_points)

				# FIXED: Transform to analysis coordinates (not screen coordinates)
				crop_region = self.current_frame_crop_region

				if crop_region:
					crop_x, crop_y, crop_w, crop_h = crop_region

					# Convert to relative coordinates within crop
					relative_x = (face_x - crop_x) / crop_w
					relative_y = (face_y - crop_y) / crop_h

					# Scale to analysis area size
					analysis_x = relative_x * self.analysis_width
					analysis_y = relative_y * self.analysis_height
				else:
					# No crop, scale directly from camera to analysis area
					analysis_x = (face_x / self.actual_camera_resolution[0]) * self.analysis_width
					analysis_y = (face_y / self.actual_camera_resolution[1]) * self.analysis_height

				# Apply mirroring (horizontal flip)
				analysis_x = self.analysis_width - analysis_x

				# FIXED: For face center, we DON'T apply rotation since analysis grid is in pre-rotation coordinates
				# Face center stays in analysis coordinates (before rotation)

				# Ensure within analysis area bounds
				if 0 <= analysis_x <= self.analysis_width and 0 <= analysis_y <= self.analysis_height:
					self.face_center = (analysis_x, analysis_y)
				else:
					self.face_center = None
			else:
				self.face_center = None

		except Exception as e:
			print(f"Error calculating face center: {e}")
			self.face_center = None

	def _transform_camera_to_final_coords(self, kp_x, kp_y):
		"""Transform camera coordinates to final screen coordinates - FIXED for analysis area"""
		# Use the same transformations as the video display
		crop_region = self.current_frame_crop_region

		if crop_region:
			crop_x, crop_y, crop_w, crop_h = crop_region

			# Convert to relative coordinates within crop
			relative_x = (kp_x - crop_x) / crop_w
			relative_y = (kp_y - crop_y) / crop_h

			# FIXED: Scale to ANALYSIS AREA SIZE, not full display
			analysis_x = relative_x * self.analysis_width
			analysis_y = relative_y * self.analysis_height
		else:
			# No crop, scale directly from camera to analysis area
			analysis_x = (kp_x / self.actual_camera_resolution[0]) * self.analysis_width
			analysis_y = (kp_y / self.actual_camera_resolution[1]) * self.analysis_height

		# Apply mirroring (horizontal flip)
		analysis_x = self.analysis_width - analysis_x

		# FIXED: Apply 90° rotation and add margins
		# After rotation: (x,y) -> (y, width-x)
		rotated_x = analysis_y
		rotated_y = self.analysis_width - analysis_x

		# Add margins to get final screen coordinates
		final_x = rotated_x + self.margin_left
		final_y = rotated_y + self.margin_top

		# Clamp to screen bounds
		final_x = max(0, min(self.display_width - 1, final_x))
		final_y = max(0, min(self.display_height - 1, final_y))

		return final_x, final_y

	def _calculate_face_shoulder_crop_region(self):
		"""Calculate crop region focusing on face and shoulders - DEBUG VERSION"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores) == 0:
			return None

		try:
			best_person_idx = np.argmax(self.last_scores)
			if best_person_idx >= len(self.last_keypoints):
				return None

			person_keypoints = self.last_keypoints[best_person_idx]
			if len(person_keypoints) < 17:
				return None

			# Face and shoulder keypoints
			target_indices = [0, 1, 2, 3, 4, 5, 6]  # Face + shoulders
			valid_points = []

			for idx in target_indices:
				if idx < len(person_keypoints):
					kp_x, kp_y, visibility = person_keypoints[idx]
					if visibility > self.visibility_threshold:
						valid_points.append((kp_x, kp_y))

			if len(valid_points) < 2:
				return None

			# Calculate bounding box
			x_coords = [p[0] for p in valid_points]
			y_coords = [p[1] for p in valid_points]

			min_x = min(x_coords)
			max_x = max(x_coords)
			min_y = min(y_coords)
			max_y = max(y_coords)

			# Add padding
			width = max_x - min_x
			height = max_y - min_y
			padding_x = width * self.crop_padding_factor
			padding_y = height * self.crop_padding_factor

			crop_x = max(0, min_x - padding_x)
			crop_y = max(0, min_y - padding_y)
			crop_w = min(self.actual_camera_resolution[0] - crop_x, width + 2 * padding_x)
			crop_h = min(self.actual_camera_resolution[1] - crop_y, height + 2 * padding_y)

			# DEBUG: Choose aspect ratio based on crop rotation setting
			if self.crop_aspect_rotated:
				# Use rotated aspect ratio (height/width) for final display after video rotation
				target_aspect = self.analysis_height / self.analysis_width
			else:
				# Use normal aspect ratio (width/height)
				target_aspect = self.analysis_width / self.analysis_height

			crop_aspect = crop_w / crop_h

			if crop_aspect > target_aspect:
				# Crop is wider than needed, adjust height
				new_crop_h = crop_w / target_aspect
				height_diff = new_crop_h - crop_h
				crop_y = max(0, crop_y - height_diff / 2)
				crop_h = min(self.actual_camera_resolution[1] - crop_y, new_crop_h)
				if crop_y + crop_h > self.actual_camera_resolution[1]:
					crop_h = self.actual_camera_resolution[1] - crop_y
					crop_w = crop_h * target_aspect
					width_diff = (min(self.actual_camera_resolution[0] - crop_x, crop_w) - crop_w) / 2
					crop_x = max(0, crop_x + width_diff)
			else:
				# Crop is taller than needed, adjust width
				new_crop_w = crop_h * target_aspect
				width_diff = new_crop_w - crop_w
				crop_x = max(0, crop_x - width_diff / 2)
				crop_w = min(self.actual_camera_resolution[0] - crop_x, new_crop_w)
				if crop_x + crop_w > self.actual_camera_resolution[0]:
					crop_w = self.actual_camera_resolution[0] - crop_x
					crop_h = crop_w / target_aspect
					height_diff = (min(self.actual_camera_resolution[1] - crop_y, crop_h) - crop_h) / 2
					crop_y = max(0, crop_y + height_diff)

			return (int(crop_x), int(crop_y), int(crop_w), int(crop_h))

		except Exception as e:
			print(f"Error calculating crop region: {e}")
			return None

	def generate_analysis_grid(self):
		"""Generate adaptive grid with mixed nesting - more organic/random appearance"""
		self.analysis_areas = []

		# Start with the largest grid (152x152) and subdivide with variety
		large_step = self.background_grid_size  # 152px

		y = 0
		row_count = 0
		while y < self.analysis_height:
			x = 0
			col_count = 0
			while x < self.analysis_width:
				# Check what grid size this area should use
				grid_type = self._get_grid_type_for_area(x, y, large_step)

				if grid_type == 'background':
					# Use full 152x152 area
					actual_width = min(large_step, self.analysis_width - x)
					actual_height = min(large_step, self.analysis_height - y)

					if actual_width > 0 and actual_height > 0:
						area_info = {
							'x': x + self.margin_left,
							'y': y + self.margin_top,
							'width': actual_width,
							'height': actual_height,
							'grid_size': large_step,
							'analysis_x': x,
							'analysis_y': y,
							'row': row_count,
							'col': col_count,
							'type': 'background'
						}
						self.analysis_areas.append(area_info)

				elif grid_type == 'shoulder':
					# MIXED: Some medium areas, some medium split into small
					medium_step = self.shoulder_grid_size  # 76px
					small_step = self.face_grid_size      # 38px

					for dy in range(0, large_step, medium_step):
						for dx in range(0, large_step, medium_step):
							sub_x = x + dx
							sub_y = y + dy

							if sub_x < self.analysis_width and sub_y < self.analysis_height:
								# Randomly decide: keep as medium or split to small
								# More likely to stay medium in shoulder areas
								split_to_small = (col_count + row_count + dx//medium_step + dy//medium_step) % 3 == 0

								if split_to_small:
									# Split this 76x76 into 4 small 38x38 areas
									for sdy in range(0, medium_step, small_step):
										for sdx in range(0, medium_step, small_step):
											ssub_x = sub_x + sdx
											ssub_y = sub_y + sdy

											if ssub_x < self.analysis_width and ssub_y < self.analysis_height:
												actual_width = min(small_step, self.analysis_width - ssub_x)
												actual_height = min(small_step, self.analysis_height - ssub_y)

												if actual_width > 0 and actual_height > 0:
													area_info = {
														'x': ssub_x + self.margin_left,
														'y': ssub_y + self.margin_top,
														'width': actual_width,
														'height': actual_height,
														'grid_size': small_step,
														'analysis_x': ssub_x,
														'analysis_y': ssub_y,
														'row': row_count,
														'col': col_count,
														'type': 'face'  # Small areas in shoulder zone
													}
													self.analysis_areas.append(area_info)
								else:
									# Keep as medium 76x76 area
									actual_width = min(medium_step, self.analysis_width - sub_x)
									actual_height = min(medium_step, self.analysis_height - sub_y)

									if actual_width > 0 and actual_height > 0:
										area_info = {
											'x': sub_x + self.margin_left,
											'y': sub_y + self.margin_top,
											'width': actual_width,
											'height': actual_height,
											'grid_size': medium_step,
											'analysis_x': sub_x,
											'analysis_y': sub_y,
											'row': row_count,
											'col': col_count,
											'type': 'shoulder'
										}
										self.analysis_areas.append(area_info)

				elif grid_type == 'face':
					# MIXED: Some medium areas, mostly small areas
					medium_step = self.shoulder_grid_size  # 76px
					small_step = self.face_grid_size      # 38px

					for dy in range(0, large_step, medium_step):
						for dx in range(0, large_step, medium_step):
							sub_x = x + dx
							sub_y = y + dy

							if sub_x < self.analysis_width and sub_y < self.analysis_height:
								# Mostly split to small, but occasionally keep medium
								keep_as_medium = (col_count + row_count + dx//medium_step + dy//medium_step) % 5 == 0

								if keep_as_medium:
									# Keep as medium 76x76 area occasionally
									actual_width = min(medium_step, self.analysis_width - sub_x)
									actual_height = min(medium_step, self.analysis_height - sub_y)

									if actual_width > 0 and actual_height > 0:
										area_info = {
											'x': sub_x + self.margin_left,
											'y': sub_y + self.margin_top,
											'width': actual_width,
											'height': actual_height,
											'grid_size': medium_step,
											'analysis_x': sub_x,
											'analysis_y': sub_y,
											'row': row_count,
											'col': col_count,
											'type': 'shoulder'  # Medium areas in face zone
										}
										self.analysis_areas.append(area_info)
								else:
									# Split this 76x76 into 4 small 38x38 areas (most common)
									for sdy in range(0, medium_step, small_step):
										for sdx in range(0, medium_step, small_step):
											ssub_x = sub_x + sdx
											ssub_y = sub_y + sdy

											if ssub_x < self.analysis_width and ssub_y < self.analysis_height:
												actual_width = min(small_step, self.analysis_width - ssub_x)
												actual_height = min(small_step, self.analysis_height - ssub_y)

												if actual_width > 0 and actual_height > 0:
													area_info = {
														'x': ssub_x + self.margin_left,
														'y': ssub_y + self.margin_top,
														'width': actual_width,
														'height': actual_height,
														'grid_size': small_step,
														'analysis_x': ssub_x,
														'analysis_y': ssub_y,
														'row': row_count,
														'col': col_count,
														'type': 'face'
													}
													self.analysis_areas.append(area_info)

				x += large_step
				col_count += 1
			y += large_step
			row_count += 1

		# Enhanced debug output
		if self.debug_mode and self.frame_count % 60 == 0:
			print(f"DEBUG: Generated {len(self.analysis_areas)} grid areas in {row_count} rows")
			print(f"DEBUG: TV dimensions: {self.display_width}x{self.display_height}")
			print(f"DEBUG: Analysis dimensions: {self.analysis_width}x{self.analysis_height}")
			print(f"DEBUG: Face center: {self.face_center}")

			if len(self.analysis_areas) > 0:
				# Check grid density
				face_areas = len([a for a in self.analysis_areas if a['type'] == 'face'])
				shoulder_areas = len([a for a in self.analysis_areas if a['type'] == 'shoulder'])
				background_areas = len([a for a in self.analysis_areas if a['type'] == 'background'])
				print(f"DEBUG: Mixed grid - Face: {face_areas}, Shoulder: {shoulder_areas}, Background: {background_areas}")

				# Show size distribution
				size_38 = len([a for a in self.analysis_areas if a['grid_size'] == 38])
				size_76 = len([a for a in self.analysis_areas if a['grid_size'] == 76])
				size_152 = len([a for a in self.analysis_areas if a['grid_size'] == 152])
				print(f"DEBUG: Size distribution - 38px: {size_38}, 76px: {size_76}, 152px: {size_152}")

	def _get_grid_type_for_area(self, x, y, area_size):
		"""Determine grid type for a large area based on face center"""
		if self.face_center is None:
			return 'background'

		face_x, face_y = self.face_center

		# Check if any part of this area is within face/shoulder distance
		# Test center and corners of the area
		test_points = [
			(x + area_size//2, y + area_size//2),  # center
			(x, y),  # top-left
			(x + area_size, y),  # top-right
			(x, y + area_size),  # bottom-left
			(x + area_size, y + area_size)  # bottom-right
		]

		min_distance = float('inf')
		for test_x, test_y in test_points:
			if 0 <= test_x <= self.analysis_width and 0 <= test_y <= self.analysis_height:
				distance = np.sqrt((test_x - face_x) ** 2 + (test_y - face_y) ** 2)
				min_distance = min(min_distance, distance)

		# Use smaller thresholds since we're checking area coverage
		if min_distance <= 100:  # Face area
			return 'face'
		elif min_distance <= 200:  # Shoulder area
			return 'shoulder'
		else:
			return 'background'

	def apply_threshold_effect(self, surface):
		"""Apply two-color threshold effect to surface"""
		try:
			# Convert surface to array
			surface_array = pygame.surfarray.array3d(surface)

			# Convert to grayscale
			gray = np.dot(surface_array, [0.299, 0.587, 0.114])

			# Apply threshold
			binary = np.where(gray > self.threshold_value, 255, 0)

			# Create RGB image (white/black)
			thresholded = np.stack([binary, binary, binary], axis=2).astype(np.uint8)

			# Convert back to surface
			result_surface = pygame.surfarray.make_surface(thresholded)
			return result_surface

		except Exception as e:
			print(f"Error applying threshold: {e}")
			return surface

	def draw_video_display(self):
		"""Draw the main video display - FIXED: Video fills analysis area properly"""
		self.update_fps()
		self.screen.fill(self.black)

		if self.current_frame is None:
			font = pygame.font.Font(None, 74)
			text = font.render("Initializing camera...", True, self.white)
			# Rotate text for physically rotated TV
			text = pygame.transform.rotate(text, 90)
			text_rect = text.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(text, text_rect)
			return

		try:
			# Convert frame to pygame surface
			frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
			frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

			# FIXED: Crop and scale to fill the ENTIRE ANALYSIS AREA
			crop_region = self.current_frame_crop_region
			if crop_region and self.pose_detected:
				crop_x, crop_y, crop_w, crop_h = crop_region
				if crop_w > 0 and crop_h > 0:
					cropped_surface = pygame.Surface((crop_w, crop_h))
					cropped_surface.blit(frame_surface, (0, 0), (crop_x, crop_y, crop_w, crop_h))
					# Scale cropped area to fill analysis area
					display_surface = pygame.transform.scale(cropped_surface, (self.analysis_width, self.analysis_height))
				else:
					# Scale full frame to fill analysis area
					display_surface = pygame.transform.scale(frame_surface, (self.analysis_width, self.analysis_height))
			else:
				# No crop, scale full frame to fill analysis area
				display_surface = pygame.transform.scale(frame_surface, (self.analysis_width, self.analysis_height))

			# DEBUG: Apply horizontal flip if enabled
			if self.video_flip_h:
				display_surface = pygame.transform.flip(display_surface, True, False)

			# DEBUG: Apply rotation if enabled
			if self.video_rotate_90:
				display_surface = pygame.transform.rotate(display_surface, 90)

			# Apply two-color threshold effect
			display_surface = self.apply_threshold_effect(display_surface)

			# FIXED: Draw video positioned to fill analysis area
			# After rotation: width/height are swapped, so it's analysis_height x analysis_width
			self.screen.blit(display_surface, (self.margin_left, self.margin_top))

			# Generate and draw analysis grid (in TV coordinates)
			self.generate_analysis_grid()
			self.draw_analysis_grid()

			# Draw debug information if enabled
			if self.debug_mode:
				self.draw_pose_debug_overlay()
				self.draw_status_overlay()

		except Exception as e:
			print(f"Error drawing video: {e}")
			font = pygame.font.Font(None, 48)
			text = font.render(f"Video error: {str(e)}", True, self.red)
			text = pygame.transform.rotate(text, 90)  # Rotate for TV
			text_rect = text.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(text, text_rect)

	def draw_analysis_grid(self):
		"""Draw 1px red squares over all analysis areas"""
		# Draw all grid areas
		for i, area in enumerate(self.analysis_areas):
			# Use different colors for different grid types when in debug mode
			if self.debug_mode:
				if area['type'] == 'face':
					color = (255, 100, 100)  # Light red
				elif area['type'] == 'shoulder':
					color = (255, 200, 100)  # Orange
				else:
					color = (100, 100, 255)  # Light blue
			else:
				color = self.red

			# Draw 1px border
			pygame.draw.rect(self.screen, color,
							(area['x'], area['y'], area['width'], area['height']), 1)

			# Draw area number in debug mode (for first few areas)
			if self.debug_mode and i < 20:
				font = pygame.font.Font(None, 24)
				text = font.render(str(i), True, self.white)
				text_rect = text.get_rect()
				if text_rect.width < area['width'] and text_rect.height < area['height']:
					self.screen.blit(text, (area['x'] + 2, area['y'] + 2))

		# Draw analysis area boundary in debug mode
		if self.debug_mode:
			# Draw the full analysis area boundary in bright blue
			analysis_rect = (self.margin_left, self.margin_top, self.analysis_width, self.analysis_height)
			pygame.draw.rect(self.screen, (0, 255, 255), analysis_rect, 3)  # Cyan border

			# Draw margin areas to show what's excluded
			# Top margin
			pygame.draw.rect(self.screen, (255, 0, 255), (0, 0, self.display_width, self.margin_top), 1)
			# Left margin
			pygame.draw.rect(self.screen, (255, 0, 255), (0, 0, self.margin_left, self.display_height), 1)
			# Right margin
			pygame.draw.rect(self.screen, (255, 0, 255), (self.display_width - self.margin_right, 0, self.margin_right, self.display_height), 1)
			# Bottom margin
			pygame.draw.rect(self.screen, (255, 0, 255), (0, self.display_height - self.margin_bottom, self.display_width, self.margin_bottom), 1)

	def draw_pose_debug_overlay(self):
		"""Draw pose keypoints and connections for debugging - FIXED coordinate transformation"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores) == 0:
			return

		try:
			best_person_idx = np.argmax(self.last_scores)
			if best_person_idx >= len(self.last_keypoints):
				return

			person_keypoints = self.last_keypoints[best_person_idx]
			if len(person_keypoints) < 17:
				return

			# Convert keypoints to final screen coordinates using the SAME transformation pipeline
			screen_points = {}
			target_indices = [0, 1, 2, 3, 4, 5, 6]  # Face + shoulders

			for idx in target_indices:
				if idx < len(person_keypoints):
					kp_x, kp_y, visibility = person_keypoints[idx]
					if visibility > self.visibility_threshold:
						# FIXED: Use the same transformation pipeline as video
						final_x, final_y = self._transform_camera_to_final_coords(kp_x, kp_y)
						screen_points[idx] = (int(final_x), int(final_y))

			# Draw connections
			face_connections = [(0, 1), (0, 2), (1, 3), (2, 4)]
			shoulder_connections = [(5, 6)]

			for start_idx, end_idx in face_connections:
				if start_idx in screen_points and end_idx in screen_points:
					pygame.draw.line(self.screen, self.white,
								   screen_points[start_idx], screen_points[end_idx], 2)

			for start_idx, end_idx in shoulder_connections:
				if start_idx in screen_points and end_idx in screen_points:
					pygame.draw.line(self.screen, self.green,
								   screen_points[start_idx], screen_points[end_idx], 2)

			# Draw keypoints
			for idx in screen_points:
				color = self.red if idx == 0 else self.white if idx <= 4 else self.green
				pygame.draw.circle(self.screen, color, screen_points[idx], 6)

			# FIXED: Draw face center in correct coordinates
			if self.face_center:
				# Face center is in analysis coordinates, convert to screen coordinates
				face_screen_x = self.face_center[0] + self.margin_left
				face_screen_y = self.face_center[1] + self.margin_top

				# Draw a yellow circle to mark face center
				pygame.draw.circle(self.screen, (255, 255, 0),
								 (int(face_screen_x), int(face_screen_y)), 10, 2)

				# Debug: print face center coordinates occasionally
				if self.debug_mode and self.frame_count % 60 == 0:
					print(f"DEBUG: Face center in analysis coords: {self.face_center}")
					print(f"DEBUG: Face center in screen coords: ({face_screen_x}, {face_screen_y})")
					print(f"DEBUG: Display size: {self.display_width}x{self.display_height}")
					print(f"DEBUG: Analysis area size: {self.analysis_width}x{self.analysis_height}")

		except Exception as e:
			print(f"Error drawing pose debug overlay: {e}")

	def draw_status_overlay(self):
		"""Draw FPS and stats overlay - ROTATED for physically rotated TV"""
		font = pygame.font.Font(None, 48)

		# Create text surfaces
		fps_text = f"FPS: {self.current_fps:.1f}"
		fps_surface = font.render(fps_text, True, self.green)

		# Grid statistics
		if hasattr(self, 'analysis_areas'):
			face_areas = len([a for a in self.analysis_areas if a['type'] == 'face'])
			shoulder_areas = len([a for a in self.analysis_areas if a['type'] == 'shoulder'])
			background_areas = len([a for a in self.analysis_areas if a['type'] == 'background'])

			stats_text = f"Grid: F{face_areas} S{shoulder_areas} B{background_areas}"
			stats_surface = font.render(stats_text, True, self.green)
		else:
			stats_surface = font.render("Grid: Loading...", True, self.green)

		# Threshold value display
		threshold_text = f"Threshold: {self.threshold_value}"
		threshold_surface = font.render(threshold_text, True, self.green)

		# Crop region info
		if self.current_frame_crop_region:
			crop_x, crop_y, crop_w, crop_h = self.current_frame_crop_region
			crop_text = f"Crop: {crop_w}x{crop_h}"
		else:
			crop_text = "Crop: Full frame"
		crop_surface = font.render(crop_text, True, self.green)

		# ROTATE text 90° for physically rotated TV
		fps_surface = pygame.transform.rotate(fps_surface, 90)
		stats_surface = pygame.transform.rotate(stats_surface, 90)
		threshold_surface = pygame.transform.rotate(threshold_surface, 90)
		crop_surface = pygame.transform.rotate(crop_surface, 90)

		# Position for rotated TV - right side becomes bottom when TV is rotated
		base_x = self.display_width - 100  # Right side of landscape TV

		self.screen.blit(fps_surface, (base_x, 20))
		self.screen.blit(stats_surface, (base_x - 60, 20))
		self.screen.blit(threshold_surface, (base_x - 120, 20))
		self.screen.blit(crop_surface, (base_x - 180, 20))

		# Debug coordinate info
		if self.frame_count % 60 == 0:
			print(f"DEBUG: FPS overlay position: ({base_x}, 20) on screen {self.display_width}x{self.display_height}")
			print(f"DEBUG: Current FPS: {self.current_fps:.1f}")

	def prepare_analysis_areas(self):
		"""Prepare image areas for analysis (placeholder for future processing)"""
		prepared_areas = []

		for area in self.analysis_areas:
			area_data = {
				'position': (area['x'], area['y']),
				'size': (area['width'], area['height']),
				'type': area['type'],
				'grid_size': area['grid_size'],
				# Future: 'image_data': extracted_image_array,
				# Future: 'analysis_result': processed_data
			}
			prepared_areas.append(area_data)

		return prepared_areas

	def handle_events(self):
		"""Handle pygame events"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					return False
				elif event.key == pygame.K_d:
					self.debug_mode = not self.debug_mode
					print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
				elif event.key == pygame.K_t:
					# Adjust threshold value
					self.threshold_value = (self.threshold_value + 32) % 256
					print(f"Threshold value: {self.threshold_value}")
				# DEBUG KEYS FOR ORIENTATION TESTING
				elif event.key == pygame.K_1:
					# Video rotation should stay True - this is just for testing
					self.video_rotate_90 = not self.video_rotate_90
					print(f"Video rotate 90°: {self.video_rotate_90} (should stay True)")
				elif event.key == pygame.K_2:
					# Video flip should stay True - this is just for testing
					self.video_flip_h = not self.video_flip_h
					print(f"Video flip horizontal: {self.video_flip_h} (should stay True)")
				elif event.key == pygame.K_3:
					# Toggle crop aspect ratio calculation
					self.crop_aspect_rotated = not self.crop_aspect_rotated
					print(f"Crop uses rotated aspect ratio: {self.crop_aspect_rotated}")
				elif event.key == pygame.K_4:
					# Toggle pose rotation
					self.pose_rotate_90 = not self.pose_rotate_90
					print(f"Pose rotate 90°: {self.pose_rotate_90}")
				elif event.key == pygame.K_5:
					# Toggle pose horizontal flip
					self.pose_flip_h = not self.pose_flip_h
					print(f"Pose flip horizontal: {self.pose_flip_h}")
				elif event.key == pygame.K_0:
					# Log current combination
					print(f"\n=== CURRENT ORIENTATION COMBINATION ===")
					print(f"Video rotate 90°: {self.video_rotate_90}")
					print(f"Video flip horizontal: {self.video_flip_h}")
					print(f"Crop uses rotated aspect: {self.crop_aspect_rotated}")
					print(f"Pose rotate 90°: {self.pose_rotate_90}")
					print(f"Pose flip horizontal: {self.pose_flip_h}")
					print(f"=======================================\n")

		return True

	def run(self):
		"""Main application loop"""
		running = True

		print("Controls:")
		print("- ESC: Exit")
		print("- 'd': Toggle debug mode (pose keypoints + FPS)")
		print("- 't': Adjust threshold value for two-color effect")
		print("=== DEBUG ORIENTATION CONTROLS ===")
		print("- '1': Toggle video rotation (keep True)")
		print("- '2': Toggle video flip (keep True)")
		print("- '3': Toggle crop aspect ratio calculation")
		print("- '4': Toggle pose rotation 90°")
		print("- '5': Toggle pose horizontal flip")
		print("- '0': Log current combination")
		print("===================================")
		print(f"TV Display: {self.display_width}x{self.display_height} (landscape)")
		print(f"Analysis grid: {self.analysis_width}x{self.analysis_height} with adaptive sizing")
		print(f"TV coordinates: 0,0 to {self.display_width},{self.display_height}")
		print(f"Analysis coordinates: {self.margin_left},{self.margin_top} to {self.margin_left + self.analysis_width},{self.margin_top + self.analysis_height}")

		# Force debug mode on initially
		self.debug_mode = True
		print("DEBUG MODE ENABLED BY DEFAULT")

		# Show initial orientation settings
		print(f"\n=== INITIAL ORIENTATION SETTINGS ===")
		print(f"Video rotate 90°: {self.video_rotate_90}")
		print(f"Video flip horizontal: {self.video_flip_h}")
		print(f"Crop uses rotated aspect: {self.crop_aspect_rotated}")
		print(f"Pose rotate 90°: {self.pose_rotate_90}")
		print(f"Pose flip horizontal: {self.pose_flip_h}")
		print(f"====================================\n")

		while running:
			running = self.handle_events()
			self.draw_video_display()

			# Prepare analysis areas (for future use)
			if hasattr(self, 'analysis_areas') and self.analysis_areas:
				prepared = self.prepare_analysis_areas()

			pygame.display.flip()
			self.clock.tick(30)

		# Cleanup
		try:
			if hasattr(self, 'picam2') and self.picam2:
				self.picam2.stop()
		except Exception as e:
			print(f"Warning: Error stopping camera: {e}")

		try:
			pygame.quit()
		except Exception as e:
			print(f"Warning: Error closing pygame: {e}")

		print("Application closed successfully")

if __name__ == "__main__":
	try:
		app = FaceAnalysisApp()
		app.run()
	except KeyboardInterrupt:
		print("\nExiting...")
	except Exception as e:
		print(f"Error: {e}")
		print("Make sure you're running on a Raspberry Pi with IMX500 camera")
		print("Required model: /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")