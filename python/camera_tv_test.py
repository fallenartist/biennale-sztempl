#!/usr/bin/env python3
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
		self.camera_input_size = (640, 480)  # IMX500 input resolution
		self.pose_model_path = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

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

		# Window size for pose estimation (matching demo)
		self.WINDOW_SIZE_H_W = (480, 640)

		# Initialize IMX500 camera with pose estimation
		self.setup_imx500_camera()

		self.clock = pygame.time.Clock()
		self.debug_mode = False

		print(f"Display setup: {self.display_width}x{self.display_height}")
		print(f"Rotation: {'90° (vertical TV)' if self.rotate_90_degrees else 'none (horizontal)'}")

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

			# Set defaults
			if self.intrinsics.inference_rate is None:
				self.intrinsics.inference_rate = 15  # Slightly higher for smoother video

			self.intrinsics.update_with_defaults()

			# Initialize camera
			self.picam2 = Picamera2(self.imx500.camera_num)
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_input_size, "format": "RGB888"},
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

		except Exception as e:
			print(f"Failed to initialize IMX500: {e}")
			print("Falling back to standard camera without pose estimation")
			# Fallback to regular camera
			self.picam2 = Picamera2()
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_input_size, "format": "RGB888"}
			)
			self.picam2.configure(config)
			self.picam2.start()
			self.imx500 = None

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
				else:
					self.pose_detected = False
			else:
				self.pose_detected = False

		except Exception as e:
			print(f"Error parsing pose results: {e}")
			self.pose_detected = False

	def get_face_shoulder_crop_region(self):
		"""Calculate crop region focusing only on face and shoulders with smaller margins"""
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

			# Add smaller margins as requested
			width = max_x - min_x
			height = max_y - min_y

			# Apply smaller padding
			padding_x = width * self.crop_padding_factor
			padding_y = height * self.crop_padding_factor

			# Calculate crop region with smaller margins
			crop_x = max(0, min_x - padding_x)
			crop_y = max(0, min_y - padding_y)
			crop_w = min(self.camera_input_size[0] - crop_x, width + 2 * padding_x)
			crop_h = min(self.camera_input_size[1] - crop_y, height + 2 * padding_y)

			return (int(crop_x), int(crop_y), int(crop_w), int(crop_h))

		except Exception as e:
			print(f"Error calculating crop region: {e}")
			return None

	def draw_video_display(self):
		"""Draw the main video display with face/shoulder cropping"""
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
			frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

			# Get crop region if pose detected
			crop_region = self.get_face_shoulder_crop_region()

			if crop_region and self.pose_detected:
				# Crop to face and shoulders area
				crop_x, crop_y, crop_w, crop_h = crop_region

				# Create cropped surface
				cropped_surface = pygame.Surface((crop_w, crop_h))
				cropped_surface.blit(frame_surface, (0, 0), (crop_x, crop_y, crop_w, crop_h))

				# Scale cropped area to fill display
				if crop_w > 0 and crop_h > 0:
					display_surface = pygame.transform.scale(cropped_surface, (self.display_width, self.display_height))
				else:
					display_surface = pygame.transform.scale(frame_surface, (self.display_width, self.display_height))
			else:
				# No pose detected, show full frame scaled
				display_surface = pygame.transform.scale(frame_surface, (self.display_width, self.display_height))

			# Apply rotation if needed
			if self.rotate_90_degrees:
				display_surface = pygame.transform.rotate(display_surface, 90)

			# Draw to screen
			self.screen.blit(display_surface, (0, 0))

			# Draw status overlay
			self.draw_status_overlay()

		except Exception as e:
			print(f"Error drawing video: {e}")
			# Show error message
			font = pygame.font.Font(None, 48)
			text = font.render(f"Video error: {str(e)}", True, self.red)
			text_rect = text.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(text, text_rect)

	def draw_status_overlay(self):
		"""Draw status information overlay"""
		font = pygame.font.Font(None, 48)

		# Pose detection status
		status_text = "FACE & SHOULDERS DETECTED" if self.pose_detected else "SEARCHING FOR PERSON..."
		status_color = self.green if self.pose_detected else self.white

		text = font.render(status_text, True, status_color)

		# Position in corner (adjust for rotation)
		if self.rotate_90_degrees:
			text_pos = (20, self.display_height - 60)
		else:
			text_pos = (20, 20)

		self.screen.blit(text, text_pos)

		# Debug mode indicator
		if self.debug_mode:
			debug_text = font.render("DEBUG MODE", True, self.red)
			if self.rotate_90_degrees:
				debug_pos = (20, self.display_height - 120)
			else:
				debug_pos = (20, 80)
			self.screen.blit(debug_text, debug_pos)

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

		return True

	def run(self):
		"""Main application loop"""
		running = True

		print("Controls:")
		print("- ESC: Exit")
		print("- 'd': Toggle debug mode")
		print("- 'r': Toggle 90° rotation for vertical TV")
		print(f"IMX500 status: {'Available' if self.imx500 else 'Not available'}")
		print(f"Target: Face and shoulders detection at {self.display_width}x{self.display_height}")

		while running:
			running = self.handle_events()

			self.draw_video_display()

			pygame.display.flip()
			self.clock.tick(30)  # 30 FPS for smooth video

		# Cleanup
		if self.picam2:
			self.picam2.stop()
		pygame.quit()

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