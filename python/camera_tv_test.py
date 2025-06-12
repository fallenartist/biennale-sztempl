def update_fps(self):
		"""Update FPS calculation"""
		import time
		current_time = time.time()
		self.frame_times = getattr(self, 'frame_times', [])
		self.frame_times.append(current_time)
		self.frame_count = getattr(self, 'frame_count', 0) + 1

		# Keep only recent frame times (last 30 frames)
		fps_update_interval = 30
		if len(self.frame_times) > fps_update_interval:
			self.frame_times.pop(0)

		# Calculate FPS every few frames
		if self.frame_count % 10 == 0 and len(self.frame_times) > 1:
			time_span = self.frame_times[-1] - self.frame_times[0]
			if time_span > 0:
				self.current_fps = (len(self.frame_times) - 1) / time_span
			else:
				self.current_fps = 0.0
		elif not hasattr(self, 'current_fps'):
			self.current_fps = 0.0#!/usr/bin/env python3
"""
Simplified Face & Shoulders Detection
Based on biennale_draft.py - removed PRNG/tiles, focused on pose detection
Shows cropped face/shoulders video stream at full HD
"""

import numpy as np
import pygame
import cv2
import time
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
import sys

class FaceShoulderApp:
	def __init__(self):
		# ================================================
		# CONFIGURABLE VARIABLES
		# ================================================

		# Display settings
		self.fullhd_width = 1920
		self.fullhd_height = 1080
		self.rotate_90_degrees = True  # Set to True for vertical TV orientation
		self.mirror_horizontally = True  # Set to True to mirror image horizontally

		# If rotated, swap width/height for display
		if self.rotate_90_degrees:
			self.display_width = self.fullhd_height  # 1080
			self.display_height = self.fullhd_width  # 1920
		else:
			self.display_width = self.fullhd_width   # 1920
			self.display_height = self.fullhd_height # 1080

		# Pose detection thresholds
		self.visibility_threshold = 0.3      # Minimum keypoint visibility (0.0-1.0)
		self.detection_threshold = 0.3       # Minimum person confidence (0.0-1.0)

		# Crop margins (smaller as requested)
		self.face_margin = 50               # Pixels around face area
		self.shoulder_margin = 30           # Pixels around shoulder area
		self.crop_padding_factor = 0.2      # 20% padding around detected area

		# Camera settings
		self.camera_resolution = (2028, 1520)  # Actual camera capture resolution (W, H) - can be changed
		self.pose_model_path = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

		# IMX500 pose estimation model expects input in (Height, Width) format
		# This should match the camera resolution but in H,W format for the model
		self.WINDOW_SIZE_H_W = (self.camera_resolution[1], self.camera_resolution[0])  # Auto-match camera resolution

		# FPS tracking
		self.frame_times = []
		self.fps_update_interval = 30  # Update FPS every 30 frames
		self.frame_count = 0
		self.current_fps = 0.0

		# Colors
		self.black = (0, 0, 0)
		self.white = (255, 255, 255)
		self.green = (0, 255, 0)
		self.red = (255, 0, 0)

		# ================================================
		# END CONFIGURABLE VARIABLES

		# Initialize Pygame
		pygame.init()

		# Set fullscreen mode
		self.screen = pygame.display.set_mode((self.display_width, self.display_height), pygame.FULLSCREEN)
		pygame.display.set_caption("Face & Shoulders Detection - Full HD")

		# Global variables for pose data
		self.last_keypoints = None
		self.last_scores = None
		self.last_boxes = None
		self.pose_detected = False
		self.current_frame = None

		# Initialize IMX500 camera with pose estimation
		self.setup_imx500_camera()

		self.clock = pygame.time.Clock()
		self.debug_mode = False

		print(f"Display setup: {self.display_width}x{self.display_height}")
		print(f"Rotation: {'90° (vertical TV)' if self.rotate_90_degrees else 'none (horizontal)'}")
		print(f"Mirror: {'enabled' if self.mirror_horizontally else 'disabled'}")

	def setup_imx500_camera(self):
		"""Initialize IMX500 camera with pose estimation model"""
		try:
			# Use the HigherHRNet pose estimation model
			self.imx500 = IMX500(self.pose_model_path)
			self.intrinsics = self.imx500.network_intrinsics

			if not self.intrinsics:
				self.intrinsics = NetworkIntrinsics()
				self.intrinsics.task = "pose estimation"
			elif self.intrinsics.task != "pose estimation":
				print("Warning: Network is not a pose estimation task", file=sys.stderr)

			# Set defaults - optimize for higher frame rate
			if self.intrinsics.inference_rate is None:
				self.intrinsics.inference_rate = 30  # Higher FPS for smoother video

			self.intrinsics.update_with_defaults()

			# Initialize camera with optimized settings
			self.picam2 = Picamera2(self.imx500.camera_num)

			# Use higher resolution for better quality - IMX500 supports:
			# - 4056x3040 @ 10fps (full resolution)
			# - 2028x1520 @ 40fps (2x2 binned)
			# - Custom lower resolutions at higher fps
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"},
				controls={'FrameRate': self.intrinsics.inference_rate},
				buffer_count=12
			)

			# Set up pre-callback for pose processing
			self.picam2.pre_callback = self.ai_output_tensor_parse

			# Show progress bar for network loading
			self.imx500.show_network_fw_progress_bar()

			# Start camera
			self.picam2.start(config, show_preview=False)  # No preview for fullscreen operation
			self.imx500.set_auto_aspect_ratio()

			print("IMX500 pose estimation initialized successfully")
			print(f"Camera resolution: {config['main']['size']}")
			print(f"Pose model window size: {self.WINDOW_SIZE_H_W}")
			print(f"Inference rate: {self.intrinsics.inference_rate} fps")

			# Store the actual camera resolution for coordinate calculations
			self.actual_camera_resolution = config['main']['size']

			# Update WINDOW_SIZE_H_W to match actual camera resolution
			self.WINDOW_SIZE_H_W = (self.actual_camera_resolution[1], self.actual_camera_resolution[0])

		except Exception as e:
			print(f"Failed to initialize IMX500: {e}")
			print("Falling back to standard camera without pose estimation")
			# Fallback to regular camera
			self.picam2 = Picamera2()
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"}
			)
			self.picam2.configure(config)
			self.picam2.start()
			self.imx500 = None
			# Store the actual camera resolution for coordinate calculations
			self.actual_camera_resolution = self.camera_resolution
			# Update WINDOW_SIZE_H_W to match actual camera resolution
			self.WINDOW_SIZE_H_W = (self.actual_camera_resolution[1], self.actual_camera_resolution[0])

	def update_fps(self):
		"""Update FPS calculation"""
		import time
		current_time = time.time()
		self.frame_times.append(current_time)
		self.frame_count += 1

		# Keep only recent frame times (last 30 frames)
		if len(self.frame_times) > self.fps_update_interval:
			self.frame_times.pop(0)

		# Calculate FPS every few frames
		if self.frame_count % 10 == 0 and len(self.frame_times) > 1:
			time_span = self.frame_times[-1] - self.frame_times[0]
			if time_span > 0:
				self.current_fps = (len(self.frame_times) - 1) / time_span

	def ai_output_tensor_parse(self, request: CompletedRequest):
		"""Parse the pose estimation output tensor and capture frame"""
		# Capture the current frame
		try:
			with MappedArray(request, "main") as m:
				self.current_frame = m.array.copy()
		except:
			pass

		if not self.imx500:
			return

		try:
			# Get outputs from IMX500
			np_outputs = self.imx500.get_outputs(metadata=request.get_metadata(), add_batch=True)

			if np_outputs is not None:
				# Process using HigherHRNet postprocessing
				keypoints, scores, boxes = postprocess_higherhrnet(
					outputs=np_outputs,
					img_size=self.WINDOW_SIZE_H_W,
					img_w_pad=(0, 0),
					img_h_pad=(0, 0),
					detection_threshold=self.detection_threshold,
					network_postprocess=True
				)

				if scores is not None and len(scores) > 0:
					# Reshape keypoints to (num_people, 17, 3) format
					self.last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
					self.last_boxes = [np.array(b) for b in boxes]
					self.last_scores = scores
					self.pose_detected = True
					# Calculate crop region ONCE when pose data is updated
					self.current_frame_crop_region = self._calculate_face_shoulder_crop_region()
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

	def _calculate_face_shoulder_crop_region(self):
		"""Calculate crop region focusing only on face and shoulders with aspect ratio preservation"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores) == 0:
			return None

		try:
			# Find person with highest confidence
			best_person_idx = np.argmax(self.last_scores)

			if best_person_idx >= len(self.last_keypoints):
				return None

			person_keypoints = self.last_keypoints[best_person_idx]

			if len(person_keypoints) < 17:
				return None

			# COCO keypoint indices for face and shoulders only:
			# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
			# 5: left_shoulder, 6: right_shoulder
			target_indices = [0, 1, 2, 3, 4, 5, 6]  # Face + shoulders only

			valid_points = []
			for idx in target_indices:
				if idx < len(person_keypoints):
					kp_x, kp_y, visibility = person_keypoints[idx]
					if visibility > self.visibility_threshold:
						valid_points.append((kp_x, kp_y))

			if len(valid_points) < 2:  # Need at least 2 valid points
				return None

			# Calculate bounding box of face and shoulders only
			x_coords = [p[0] for p in valid_points]
			y_coords = [p[1] for p in valid_points]

			min_x = min(x_coords)
			max_x = max(x_coords)
			min_y = min(y_coords)
			max_y = max(y_coords)

			# Add margins as configured
			width = max_x - min_x
			height = max_y - min_y

			# Apply padding
			padding_x = width * self.crop_padding_factor
			padding_y = height * self.crop_padding_factor

			# Calculate initial crop region
			crop_x = max(0, min_x - padding_x)
			crop_y = max(0, min_y - padding_y)
			crop_w = min(self.camera_resolution[0] - crop_x, width + 2 * padding_x)
			crop_h = min(self.camera_resolution[1] - crop_y, height + 2 * padding_y)

			# Adjust crop to maintain aspect ratio without distortion
			display_aspect = self.display_width / self.display_height
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

			# Debug output
			if self.debug_mode and self.frame_count % 30 == 0:  # Print every 30 frames
				print(f"DEBUG: Original bbox: ({min_x:.1f}, {min_y:.1f}) -> ({max_x:.1f}, {max_y:.1f})")
				print(f"DEBUG: Final crop region: ({int(crop_x)}, {int(crop_y)}, {int(crop_w)}, {int(crop_h)})")
				print(f"DEBUG: Display aspect: {display_aspect:.3f}, Crop aspect: {crop_w/crop_h:.3f}")
				print(f"DEBUG: Valid points: {len(valid_points)}, Camera res: {self.camera_resolution}")

			# Debug output
			if self.debug_mode and self.frame_count % 30 == 0:  # Print every 30 frames
				print(f"DEBUG: Original bbox: ({min_x:.1f}, {min_y:.1f}) -> ({max_x:.1f}, {max_y:.1f})")
				print(f"DEBUG: Final crop region: ({int(crop_x)}, {int(crop_y)}, {int(crop_w)}, {int(crop_h)})")
				print(f"DEBUG: Display aspect: {display_aspect:.3f}, Crop aspect: {crop_w/crop_h:.3f}")
				print(f"DEBUG: Valid points: {len(valid_points)}, Camera res: {self.camera_resolution}")

			return (int(crop_x), int(crop_y), int(crop_w), int(crop_h))

		except Exception as e:
			print(f"Error calculating crop region: {e}")
			return None

	def get_face_shoulder_crop_region(self):
		"""Return the cached crop region calculated when pose data was updated"""
		return getattr(self, 'current_frame_crop_region', None)

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

	def draw_video_display(self):
		"""Draw the main video display with face/shoulder cropping"""
		# Update FPS tracking
		self.update_fps()

		# Clear screen
		self.screen.fill(self.black)

		if self.current_frame is None:
			# Show loading message
			font = pygame.font.Font(None, 74)
			text = font.render("Initializing camera...", True, self.white)
			text_rect = text.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(text, text_rect)
			return

		try:
			# Convert OpenCV frame to Pygame surface
			frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

			# DEBUG: Check if the coordinate system is wrong here
			if self.debug_mode and self.frame_count % 30 == 0:
				print(f"DEBUG: Original frame shape (OpenCV): {self.current_frame.shape}")  # Should be (768, 1024, 3)
				print(f"DEBUG: Frame RGB shape: {frame_rgb.shape}")

			frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

			if self.debug_mode and self.frame_count % 30 == 0:
				print(f"DEBUG: Final frame surface size (pygame): {frame_surface.get_size()}")  # Should be (1024, 768)

			# Get crop region - should be the same cached value for entire frame
			crop_region = self.get_face_shoulder_crop_region()

			# Store crop region for debug overlay to use the same value
			# (This should already be the same, but ensuring consistency)

			if crop_region and self.pose_detected:
				# Crop to face and shoulders area
				crop_x, crop_y, crop_w, crop_h = crop_region

				# Debug output for video cropping
				if self.debug_mode and self.frame_count % 30 == 0:
					print(f"DEBUG: Video display using crop region: {crop_region}")
					print(f"DEBUG: Frame surface size: {frame_surface.get_size()}")

				# Create cropped surface
				if crop_w > 0 and crop_h > 0:
					cropped_surface = pygame.Surface((crop_w, crop_h))
					cropped_surface.blit(frame_surface, (0, 0), (crop_x, crop_y, crop_w, crop_h))

					# Scale cropped area to fill display (already aspect-ratio corrected)
					display_surface = pygame.transform.scale(cropped_surface, (self.display_width, self.display_height))

					if self.debug_mode and self.frame_count % 30 == 0:
						print(f"DEBUG: Cropped surface size: {cropped_surface.get_size()}")
						print(f"DEBUG: Final display size: {display_surface.get_size()}")
				else:
					# Fallback to full frame with aspect ratio preserved
					display_surface = self.scale_with_aspect_ratio(frame_surface, self.display_width, self.display_height)
			else:
				# No pose detected, show full frame with aspect ratio preserved
				display_surface = self.scale_with_aspect_ratio(frame_surface, self.display_width, self.display_height)

			# Apply mirroring if needed
			if self.mirror_horizontally:
				display_surface = pygame.transform.flip(display_surface, True, False)  # Flip horizontally

			# Apply rotation if needed
			if self.rotate_90_degrees:
				display_surface = pygame.transform.rotate(display_surface, 90)

			# Draw to screen
			self.screen.blit(display_surface, (0, 0))

			# Draw debug overlay if enabled
			if self.debug_mode:
				self.draw_pose_debug_overlay()

			# Draw status overlay
			self.draw_status_overlay()

		except Exception as e:
			print(f"Error drawing video: {e}")
			# Show error message
			font = pygame.font.Font(None, 48)
			text = font.render(f"Video error: {str(e)}", True, self.red)
			text_rect = text.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(text, text_rect)

	def draw_pose_debug_overlay(self):
		"""Draw pose keypoints and connections for debugging"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores) == 0:
			return

		try:
			# Find person with highest confidence
			best_person_idx = np.argmax(self.last_scores)

			if best_person_idx >= len(self.last_keypoints):
				return

			person_keypoints = self.last_keypoints[best_person_idx]

			if len(person_keypoints) < 17:
				return

			# COCO keypoint indices we care about:
			# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
			# 5: left_shoulder, 6: right_shoulder
			keypoint_names = ['nose', 'L eye', 'R eye', 'L ear', 'R ear', 'L shoulder', 'R shoulder']
			target_indices = [0, 1, 2, 3, 4, 5, 6]

			# Get crop region once to ensure consistency between video and debug overlay
			# Use the same crop region that was calculated when pose data was updated
			crop_region = self.get_face_shoulder_crop_region()

			# Debug output for this specific crop calculation
			if self.debug_mode and crop_region and self.frame_count % 30 == 0:
				print(f"DEBUG: Debug overlay using EXACT SAME cached crop region: {crop_region}")

			# Convert keypoints to screen coordinates
			screen_points = {}
			for i, idx in enumerate(target_indices):
				if idx < len(person_keypoints):
					kp_x, kp_y, visibility = person_keypoints[idx]
					if visibility > self.visibility_threshold:
						# Debug: print raw coordinates occasionally
						if self.debug_mode and self.frame_count % 60 == 0 and idx == 0:  # Print nose coordinates every 60 frames
							print(f"DEBUG: Nose keypoint: ({kp_x:.1f}, {kp_y:.1f}) in camera space {self.camera_resolution}")
							print(f"DEBUG: Using crop region from video: {crop_region}")

						# Convert coordinates to display space
						if crop_region:
							crop_x, crop_y, crop_w, crop_h = crop_region
							# Convert to cropped coordinates first
							relative_x = (kp_x - crop_x) / crop_w
							relative_y = (kp_y - crop_y) / crop_h
							# Then scale to display size
							screen_x = relative_x * self.display_width
							screen_y = relative_y * self.display_height
						else:
							# No crop, scale directly from camera to display
							screen_x = (kp_x / self.camera_resolution[0]) * self.display_width
							screen_y = (kp_y / self.camera_resolution[1]) * self.display_height

						# Apply mirroring if enabled (before rotation)
						if self.mirror_horizontally:
							screen_x = self.display_width - screen_x

						# Apply rotation coordinates if enabled
						if self.rotate_90_degrees:
							# 90° rotation: (x,y) -> (y, display_width-x)
							rotated_x = screen_y
							rotated_y = self.display_width - screen_x
							screen_x, screen_y = rotated_x, rotated_y

						# Clamp to screen bounds
						screen_x = max(0, min(self.display_width - 1, screen_x))
						screen_y = max(0, min(self.display_height - 1, screen_y))

						screen_points[idx] = (int(screen_x), int(screen_y))

			# Draw connection lines
			line_width = 2

			# Face connections: nose to eyes to ears
			face_connections = [
				(0, 1),  # nose to left_eye
				(0, 2),  # nose to right_eye
				(1, 3),  # left_eye to left_ear
				(2, 4),  # right_eye to right_ear
			]

			# Shoulder connection
			shoulder_connections = [
				(5, 6),  # left_shoulder to right_shoulder
			]

			# Draw face connections in white
			for start_idx, end_idx in face_connections:
				if start_idx in screen_points and end_idx in screen_points:
					pygame.draw.line(self.screen, self.white,
								   screen_points[start_idx], screen_points[end_idx], line_width)

			# Draw shoulder connections in green
			for start_idx, end_idx in shoulder_connections:
				if start_idx in screen_points and end_idx in screen_points:
					pygame.draw.line(self.screen, self.green,
								   screen_points[start_idx], screen_points[end_idx], line_width)

			# Draw keypoints as circles
			keypoint_radius = 8
			for i, idx in enumerate(target_indices):
				if idx in screen_points:
					# Color code keypoints
					if idx == 0:  # nose
						color = self.red
					elif idx in [1, 2, 3, 4]:  # eyes and ears
						color = self.white
					else:  # shoulders
						color = self.green

					pygame.draw.circle(self.screen, color, screen_points[idx], keypoint_radius)

					# Draw label
					font = pygame.font.Font(None, 24)
					text = font.render(keypoint_names[i], True, color)
					label_x = screen_points[idx][0] + 12
					label_y = screen_points[idx][1] - 12
					self.screen.blit(text, (label_x, label_y))

		except Exception as e:
			print(f"Error drawing pose debug overlay: {e}")

	def draw_rotated_text(self, text, font, color, x, y):
		"""Draw text that matches the video rotation"""
		text_surface = font.render(text, True, color)
		if self.rotate_90_degrees:
			# Rotate text 90 degrees to match video rotation
			text_surface = pygame.transform.rotate(text_surface, 90)
		return text_surface

	def draw_status_overlay(self):
		"""Draw minimal status overlay - just FPS and camera resolution"""
		font = pygame.font.Font(None, 48)

		# FPS display
		fps_text = f"FPS: {getattr(self, 'current_fps', 0.0):.1f}"
		if self.rotate_90_degrees:
		   fps_surface = self.draw_rotated_text(fps_text, font, self.white, 0, 0)
		   fps_pos = (self.display_width - 20, 20)
		else:
		   fps_surface = font.render(fps_text, True, self.white)
		   fps_pos = (self.display_width - 200, 20)
		self.screen.blit(fps_surface, fps_pos)

		# Camera resolution display (always show for reference)
		if hasattr(self, 'actual_camera_resolution'):
		   res_text = f"CAM: {self.actual_camera_resolution[0]}x{self.actual_camera_resolution[1]}"
		   if self.rotate_90_degrees:
			   res_surface = self.draw_rotated_text(res_text, font, self.white, 0, 0)
			   res_pos = (self.display_width - 20, 20 + fps_surface.get_width() + 10)
		   else:
			   res_surface = font.render(res_text, True, self.white)
			   res_pos = (self.display_width - 300, 70)
		   self.screen.blit(res_surface, res_pos)

	def handle_events(self):
		"""Handle pygame events"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					# Exit on ESC
					return False
				elif event.key == pygame.K_d:
					# Toggle debug mode
					self.debug_mode = not self.debug_mode
					print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
				elif event.key == pygame.K_r:
					# Toggle rotation
					self.rotate_90_degrees = not self.rotate_90_degrees
					print(f"Rotation: {'90° (vertical)' if self.rotate_90_degrees else 'none (horizontal)'}")

					# Update display dimensions
					if self.rotate_90_degrees:
						self.display_width = self.fullhd_height
						self.display_height = self.fullhd_width
					else:
						self.display_width = self.fullhd_width
						self.display_height = self.fullhd_height

					# Clear screen to remove any text artifacts
					self.screen.fill(self.black)
					pygame.display.flip()
				elif event.key == pygame.K_m:
					# Toggle horizontal mirroring
					self.mirror_horizontally = not self.mirror_horizontally
					print(f"Mirror: {'enabled' if self.mirror_horizontally else 'disabled'}")

		return True

	def run(self):
		"""Main application loop"""
		running = True

		print("Controls:")
		print("- ESC: Exit")
		print("- 'd': Toggle debug mode (show pose keypoints and connections)")
		print("- 'r': Toggle 90° rotation for vertical TV")
		print("- 'm': Toggle horizontal mirroring")
		print(f"IMX500 status: {'Available' if self.imx500 else 'Not available'}")
		print(f"Target: Face and shoulders detection at {self.display_width}x{self.display_height}")

		while running:
			running = self.handle_events()

			self.draw_video_display()

			pygame.display.flip()
			self.clock.tick(30)  # 30 FPS for smooth video

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
		app = FaceShoulderApp()
		app.run()
	except KeyboardInterrupt:
		print("\nExiting...")
	except Exception as e:
		print(f"Error: {e}")
		print("Make sure you're running on a Raspberry Pi with IMX500 camera")
		print("Required model: /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")