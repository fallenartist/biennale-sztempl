#!/usr/bin/env python3
"""
Mejking 2021 Festival
Visual Identity
Python version with IMX500 pose estimation - COMPLETE FIXED VERSION
"""

import numpy as np
import pygame
import random
import math
import time
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
import sys

class PRNG:
	"""Pseudo-Random Number Generator with Perlin noise support"""

	def __init__(self, seed=None):
		if seed is None:
			seed = int(time.time() * 1000) % 2147483647

		self.a = 1664525
		self.c = 1013904223
		self.m32 = 0xFFFFFFFF
		self.seed = seed % 2147483647

		# seed should be the image's unique ID!

		# Initialize Perlin noise
		self.gradients = []
		step_size = 2 * math.pi / 16
		for i in range(16):
			angle = i * step_size
			self.gradients.append([math.cos(angle), math.sin(angle)])

		# Create permutation array
		part_a = list(range(256))
		part_b = list(range(256))
		self.shuffle_array(part_a)
		self.shuffle_array(part_b)

		self.p = part_a + part_b

	def next_long(self):
		self.seed = (self.seed * self.a + self.c) & self.m32
		return self.seed

	def next_int(self, max_val=None):
		if max_val is None:
			return int(self.next_long() % 2147483647)
		return self.next_int() % max_val

	def random_float(self, min_val=0, max_val=1):
		return min_val + (self.next_int() / 2147483647) * (max_val - min_val)

	def shuffle_array(self, arr):
		for i in range(len(arr) - 1, 0, -1):
			j = self.next_int(i + 1)
			arr[i], arr[j] = arr[j], arr[i]

	def perlin(self, x, y):
		# Get grid coordinates
		i = int(x) & 255
		j = int(y) & 255

		# Get gradients at four corners
		def get_grad_index(gi, gj):
			return self.p[self.p[self.p[gi] + gj]] & 15

		grad_tl = get_grad_index(i, j)
		grad_tr = get_grad_index((i + 1) & 255, j)
		grad_bl = get_grad_index(i, (j + 1) & 255)
		grad_br = get_grad_index((i + 1) & 255, (j + 1) & 255)

		# Get relative coordinates
		u = x - int(x)
		v = y - int(y)

		# Calculate dot products
		def dot_prod(grad_idx, dx, dy):
			grad = self.gradients[grad_idx]
			return grad[0] * dx + grad[1] * dy

		dot_tl = dot_prod(grad_tl, u, v)
		dot_tr = dot_prod(grad_tr, u - 1, v)
		dot_bl = dot_prod(grad_bl, u, v - 1)
		dot_br = dot_prod(grad_br, u - 1, v - 1)

		# Interpolate
		def fade(t):
			return t * t * t * (t * (t * 6 - 15) + 10)

		def lerp(a, b, t):
			return a + fade(t) * (b - a)

		top_avg = lerp(dot_tl, dot_tr, u)
		bottom_avg = lerp(dot_bl, dot_br, v)
		result = lerp(top_avg, bottom_avg, v)

		# Map to 0-1 range
		perlin_bound = 1.0 / math.sqrt(2)
		return max(0, min(1, (result + perlin_bound) / (2 * perlin_bound)))

class MejkingApp:
	def __init__(self):
		# ================================================
		# CONFIGURABLE VARIABLES - Edit these at the top
		# ================================================

		# Screen dimensions
		self.width = 600
		self.height = 800

		# Colors
		self.black = (0, 0, 0)
		self.green = (64, 128, 64)
		self.blue = (0, 126, 192)
		self.mustard = (187, 162, 24)
		self.white = (255, 255, 255)
		self.red_tile_color = (255, 100, 100)  # Color for person-influenced tiles

		# Pose detection thresholds
		self.visibility_threshold = 0.1      # Minimum keypoint visibility (0.0-1.0)
		self.detection_threshold = 0.3       # Minimum person confidence (0.0-1.0)

		# Influence areas and strengths
		self.face_influence_radius = 120     # Pixels around face center
		self.face_influence_strength = 2.0   # Multiplier for face area
		self.shoulder_influence_radius = 80  # Pixels around shoulders
		self.shoulder_influence_strength = 1.5  # Multiplier for shoulder area
		self.person_bbox_influence = 1.8     # Multiplier for whole person area
		self.bbox_width_padding = 0.5        # Expand person bbox by 50% width
		self.bbox_height_padding = 0.3       # Expand person bbox by 30% height
		self.max_influence = 3.0             # Maximum influence multiplier

		# Debug visualization settings
		self.skeleton_line_width = 4         # Width of skeleton lines
		self.skeleton_line_color = (255, 255, 255)  # White skeleton lines
		self.keypoint_circle_size = 8        # Radius of keypoint circles
		self.show_keypoint_labels = True     # Show keypoint names

		# Keypoint colors
		self.face_keypoint_color = (255, 0, 0)     # Red for face
		self.shoulder_keypoint_color = (0, 255, 0)  # Green for shoulders
		self.leg_keypoint_color = (0, 0, 255)      # Blue for legs
		self.arm_keypoint_color = (255, 255, 0)    # Yellow for arms

		# Camera and pose estimation settings
		self.camera_input_size = (640, 480)  # IMX500 input resolution
		self.pose_model_path = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

		# ================================================
		# END CONFIGURABLE VARIABLES

		# Initialize Pygame
		pygame.init()
		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption("Mejking 2021 Festival - IMX500")

		# Global variables for pose data
		self.last_keypoints = None
		self.last_scores = None
		self.last_boxes = None
		self.pose_detected = False

		# Window size for pose estimation (matching demo)
		self.WINDOW_SIZE_H_W = (480, 640)

		# Initialize IMX500 camera with pose estimation
		self.setup_imx500_camera()

		# Load tile images from /tiles folder
		self.num_images = 8
		self.images = []
		for i in range(self.num_images):
			try:
				# Load actual image files from tiles folder
				img = pygame.image.load(f"tiles/F{i:02d}.png")
				self.images.append(img)
				print(f"Loaded tiles/F{i:02d}.png")
			except (pygame.error, FileNotFoundError):
				print(f"Warning: Could not load tiles/F{i:02d}.png, using placeholder")
				# Fallback to placeholder if file doesn't exist
				img = pygame.Surface((20, 20))
				img.fill((50 + i * 20, 100 + i * 15, 150 + i * 10))
				self.images.append(img)

		# Initialize variables
		self.seed = random.randint(0, 1000000)
		self.prng = PRNG(self.seed)
		self.sizes = [20, 40, 80]
		self.min_step = min(self.sizes)
		self.max_step = max(self.sizes)

		# Initialize parameters
		self.reset_parameters()

		# Font
		self.font = pygame.font.Font(None, int(self.min_step * 0.7))

		self.clock = pygame.time.Clock()
		self.debug_mode = False

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
				self.intrinsics.inference_rate = 10

			self.intrinsics.update_with_defaults()

			# Initialize camera
			self.picam2 = Picamera2(self.imx500.camera_num)
			config = self.picam2.create_preview_configuration(
				controls={'FrameRate': self.intrinsics.inference_rate},
				buffer_count=12
			)

			# Set up pre-callback for pose processing
			self.picam2.pre_callback = self.ai_output_tensor_parse

			# Show progress bar for network loading
			self.imx500.show_network_fw_progress_bar()

			# Start camera
			self.picam2.start(config, show_preview=False)  # No preview for headless operation
			self.imx500.set_auto_aspect_ratio()

			print("IMX500 pose estimation initialized successfully")

		except Exception as e:
			print(f"Failed to initialize IMX500: {e}")
			print("Falling back to standard camera without pose estimation")
			# Fallback to regular camera
			self.picam2 = Picamera2()
			config = self.picam2.create_preview_configuration(
				main={"size": (640, 480), "format": "RGB888"}
			)
			self.picam2.configure(config)
			self.picam2.start()
			self.imx500 = None

	def ai_output_tensor_parse(self, request: CompletedRequest):
		"""Parse the pose estimation output tensor - called as pre_callback"""
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
					detection_threshold=self.detection_threshold,  # Use configurable threshold
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

	def get_pose_influence(self, x, y):
		"""Calculate how pose affects the pattern at given coordinates"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores) == 0:
			return 1.0  # No influence

		influence = 1.0

		# Process all detected people (use highest confidence person)
		if len(self.last_scores) > 0 and len(self.last_keypoints) > 0:
			try:
				# Find person with highest confidence
				best_person_idx = np.argmax(self.last_scores)

				# Ensure the index is valid
				if best_person_idx >= len(self.last_keypoints):
					return 1.0

				person_keypoints = self.last_keypoints[best_person_idx]

				# Ensure we have enough keypoints
				if len(person_keypoints) < 17:
					return 1.0

				# Calculate the bounding box of all detected keypoints to expand influence
				valid_keypoints = []
				for kp_x, kp_y, visibility in person_keypoints:
					if visibility > self.visibility_threshold:
						screen_x = kp_x * self.width / self.camera_input_size[0]
						screen_y = kp_y * self.height / self.camera_input_size[1]
						valid_keypoints.append((screen_x, screen_y))

				if valid_keypoints:
					# Calculate bounding box of person
					min_x = min(kp[0] for kp in valid_keypoints)
					max_x = max(kp[0] for kp in valid_keypoints)
					min_y = min(kp[1] for kp in valid_keypoints)
					max_y = max(kp[1] for kp in valid_keypoints)

					# Expand the bounding box to cover more area
					width_padding = (max_x - min_x) * self.bbox_width_padding
					height_padding = (max_y - min_y) * self.bbox_height_padding

					expanded_min_x = max(0, min_x - width_padding)
					expanded_max_x = min(self.width, max_x + width_padding)
					expanded_min_y = max(0, min_y - height_padding)
					expanded_max_y = min(self.height, max_y + height_padding)

					# Check if point is within expanded bounding box
					if (expanded_min_x <= x <= expanded_max_x and
						expanded_min_y <= y <= expanded_max_y):
						influence *= self.person_bbox_influence  # Use configurable influence

				# COCO keypoint indices:
				# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
				# 5: left_shoulder, 6: right_shoulder

				# Face influence (nose, eyes, ears)
				face_keypoint_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
				face_points = []

				for idx in face_keypoint_indices:
					if idx < len(person_keypoints):
						kp_x, kp_y, visibility = person_keypoints[idx]
						if visibility > self.visibility_threshold:  # Use configurable threshold
							# Direct scaling instead of using WINDOW_SIZE_H_W
							screen_x = kp_x * self.width / self.camera_input_size[0]
							screen_y = kp_y * self.height / self.camera_input_size[1]
							face_points.append((screen_x, screen_y))

				# Calculate face center and influence
				if face_points:
					face_center_x = sum(p[0] for p in face_points) / len(face_points)
					face_center_y = sum(p[1] for p in face_points) / len(face_points)

					# Distance-based influence
					dist = math.sqrt((x - face_center_x)**2 + (y - face_center_y)**2)
					if dist < self.face_influence_radius:
						influence *= self.face_influence_strength  # Use configurable values

				# Shoulder influence
				shoulder_indices = [5, 6]  # left_shoulder, right_shoulder
				for idx in shoulder_indices:
					if idx < len(person_keypoints):
						kp_x, kp_y, visibility = person_keypoints[idx]
						if visibility > self.visibility_threshold:
							screen_x = kp_x * self.width / self.camera_input_size[0]
							screen_y = kp_y * self.height / self.camera_input_size[1]

							dist = math.sqrt((x - screen_x)**2 + (y - screen_y)**2)
							if dist < self.shoulder_influence_radius:
								influence *= self.shoulder_influence_strength

			except (IndexError, TypeError, ValueError) as e:
				# Gracefully handle any array access errors
				print(f"DEBUG: Error in pose influence calculation: {e}")
				return 1.0

		return max(0.1, min(self.max_influence, influence))  # Use configurable max

	def reset_parameters(self):
		"""Reset all animation parameters"""
		random.seed(self.seed)
		self.prng = PRNG(self.seed)

		self.ratio = self.width / self.height
		self.d0 = random.uniform(100, 200)
		self.d1 = random.uniform(25, 75)
		self.dz = random.uniform(0, 100)
		self.dv = random.uniform(0, 0.1)

		# Create shuffled image indices
		self.ii = list(range(self.num_images))
		random.shuffle(self.ii)

		# Create random positions for largest pattern
		num_points = (self.width // self.max_step * self.height // self.max_step) // random.randint(8, 13) * 2
		self.rr = []
		for i in range(0, num_points, 2):
			self.rr.append(random.randint(0, self.width // self.max_step - 1))
			self.rr.append(random.randint(0, self.height // self.max_step - 1))

	def lerp_color(self, color1, color2, t):
		"""Linear interpolation between two colors"""
		return (
			int(color1[0] + (color2[0] - color1[0]) * t),
			int(color1[1] + (color2[1] - color1[1]) * t),
			int(color1[2] + (color2[2] - color1[2]) * t)
		)

	def draw_pattern(self):
		"""Draw the main pattern with pose influence"""
		# Clear screen
		self.screen.fill(self.blue)

		# Draw pattern layers
		for p, step in enumerate(self.sizes):
			for j in range(0, self.height, step):
				for i in range(0, self.width, step):
					# Calculate base noise values
					n0 = self.prng.perlin(i / (self.d0 + self.dz), j / (self.d0 + self.dz))
					n1 = self.prng.perlin(i / (self.d1 + self.dz), j / (self.d1 + self.dz))
					n = 1 - (n0 * 0.75 + n1 * 0.25)

					# Apply pose influence
					pose_influence = self.get_pose_influence(i + step/2, j + step/2)
					n = n * pose_influence
					n = max(0, min(1, n))  # Clamp to valid range

					k = int(n * self.num_images)
					k = max(0, min(k, self.num_images - 1))

					should_draw = False

					if p == 0:  # Draw all
						should_draw = True
					elif p == 1 and self.ii[k] < 4:  # Draw some shades
						should_draw = True
					elif p == 2:  # Draw at specific positions
						for r in range(0, len(self.rr), 2):
							if (i == self.rr[r] * step and
								j == self.rr[r + 1] * step):
								should_draw = True
								break

					if should_draw:
						# Check if this tile is influenced by pose detection
						if pose_influence > 1.1 and self.pose_detected:
							# Red color for tiles influenced by person detection
							color = self.lerp_color(self.blue, self.red_tile_color, self.ii[k] / 10.0)
						else:
							# Normal color
							color = self.lerp_color(self.blue, self.white, self.ii[k] / 10.0)

						pygame.draw.rect(self.screen, color, (i, j, step, step))

						# Draw pattern image (scaled)
						scaled_img = pygame.transform.scale(self.images[self.ii[k]], (step, step))
						self.screen.blit(scaled_img, (i, j))

	def draw_pose_debug(self):
		"""Draw pose keypoints and skeleton for debugging"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores) == 0:
			print("DEBUG: No pose detected or no valid data")
			return

		print(f"DEBUG: Drawing pose for {len(self.last_scores)} people")
		if len(self.last_keypoints) > 0:
			print(f"DEBUG: Keypoints shape: {self.last_keypoints.shape}")

		# COCO keypoint names
		keypoint_names = [
			'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
			'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
			'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
			'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
		]

		# COCO pose skeleton connections (pairs of keypoint indices)
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

		# Draw skeleton and keypoints for all detected people
		for person_idx in range(min(len(self.last_scores), len(self.last_keypoints))):
			try:
				person_keypoints = self.last_keypoints[person_idx]
				print(f"DEBUG: Person {person_idx} has {len(person_keypoints)} keypoints")

				# Ensure we have the expected 17 keypoints
				if len(person_keypoints) < 17:
					print(f"DEBUG: Warning - Person {person_idx} only has {len(person_keypoints)} keypoints, expected 17")
					continue

				# First pass: Convert all keypoints to screen coordinates
				screen_points = []
				for kp_idx, (kp_x, kp_y, visibility) in enumerate(person_keypoints):
					if kp_idx < len(keypoint_names):
						print(f"DEBUG: KP {kp_idx} ({keypoint_names[kp_idx]}): x={kp_x:.1f}, y={kp_y:.1f}, vis={visibility:.2f}")

					if visibility > self.visibility_threshold:
						# Scale coordinates
						screen_x = int(kp_x * self.width / self.camera_input_size[0])
						screen_y = int(kp_y * self.height / self.camera_input_size[1])

						# Clamp to screen bounds
						screen_x = max(0, min(self.width - 1, screen_x))
						screen_y = max(0, min(self.height - 1, screen_y))

						screen_points.append((screen_x, screen_y))
						print(f"DEBUG: Mapped to screen: ({screen_x}, {screen_y})")
					else:
						screen_points.append(None)  # Mark invisible keypoints as None

				valid_points = len([p for p in screen_points if p is not None])
				print(f"DEBUG: Valid screen points: {valid_points}")

				# Second pass: Draw skeleton lines FIRST (so they appear behind keypoints)
				lines_drawn = 0
				for start_idx, end_idx in skeleton:
					# ROBUST BOUNDS CHECKING
					if (start_idx < len(screen_points) and end_idx < len(screen_points) and
						start_idx < len(person_keypoints) and end_idx < len(person_keypoints) and
						start_idx >= 0 and end_idx >= 0 and
						screen_points[start_idx] is not None and screen_points[end_idx] is not None):

						try:
							start_vis = person_keypoints[start_idx][2]
							end_vis = person_keypoints[end_idx][2]

							# Draw line if both points are visible enough
							if start_vis > self.visibility_threshold and end_vis > self.visibility_threshold:
								pygame.draw.line(self.screen, self.skeleton_line_color,
											   screen_points[start_idx], screen_points[end_idx],
											   self.skeleton_line_width)
								lines_drawn += 1
								print(f"DEBUG: Drew line {start_idx}-{end_idx}: {screen_points[start_idx]} to {screen_points[end_idx]}")
						except (IndexError, TypeError) as e:
							print(f"DEBUG: Error drawing line {start_idx}-{end_idx}: {e}")
							continue

				print(f"DEBUG: Drew {lines_drawn} skeleton lines")

				# Third pass: Draw keypoints on top of skeleton
				points_drawn = 0
				for kp_idx, (kp_x, kp_y, visibility) in enumerate(person_keypoints):
					if (visibility > self.visibility_threshold and
						kp_idx < len(keypoint_names) and
						kp_idx < len(screen_points) and
						screen_points[kp_idx] is not None):

						try:
							screen_x = int(kp_x * self.width / self.camera_input_size[0])
							screen_y = int(kp_y * self.height / self.camera_input_size[1])

							# Clamp to screen bounds
							screen_x = max(0, min(self.width - 1, screen_x))
							screen_y = max(0, min(self.height - 1, screen_y))

							# Choose color based on keypoint type
							name = keypoint_names[kp_idx]
							if any(part in name for part in ['eye', 'nose', 'ear']):
								color = self.face_keypoint_color
							elif 'shoulder' in name:
								color = self.shoulder_keypoint_color
							elif any(part in name for part in ['hip', 'knee', 'ankle']):
								color = self.leg_keypoint_color
							else:
								color = self.arm_keypoint_color

							# Draw keypoint circle
							pygame.draw.circle(self.screen, color, (screen_x, screen_y), self.keypoint_circle_size)
							points_drawn += 1

							# Draw label if enabled
							if self.show_keypoint_labels:
								text = pygame.font.Font(None, 20).render(name[:4], True, color)
								self.screen.blit(text, (screen_x + 10, screen_y - 10))
						except (IndexError, TypeError, ValueError) as e:
							print(f"DEBUG: Error drawing keypoint {kp_idx}: {e}")
							continue

				print(f"DEBUG: Drew {points_drawn} keypoints")
				print("---")

			except (IndexError, TypeError) as e:
				print(f"DEBUG: Error processing person {person_idx}: {e}")
				continue

	def draw_ui(self):
		"""Draw the user interface elements"""
		# Draw title bar
		pygame.draw.rect(self.screen, self.white,
						(0, self.min_step * 2, self.min_step * 6, self.min_step))

		title_text = self.font.render("mejking 2021", True, self.black)
		self.screen.blit(title_text, (self.min_step // 2, self.min_step * 2))

		# Draw seed bar with pose status
		pygame.draw.rect(self.screen, self.white,
						(0, self.height - self.min_step, self.width, self.min_step))

		# Show pose detection status
		pose_status = "POSE" if self.pose_detected else "----"
		debug_status = " [DBG]" if self.debug_mode else ""
		seed_text = self.font.render(f"#{self.seed} [{pose_status}]{debug_status}", True, self.black)
		self.screen.blit(seed_text, (self.min_step // 2,
								   self.height - self.min_step + self.min_step // 8))

	def update(self):
		"""Update animation parameters"""
		self.dz += self.dv

	def handle_events(self):
		"""Handle pygame events"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_d:
					# Toggle debug mode (show pose keypoints)
					self.debug_mode = not self.debug_mode
					print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
				else:
					# Reset on other key press
					self.seed = random.randint(0, 1000000)
					self.reset_parameters()
					print(f"New seed: {self.seed}")
		return True

	def run(self):
		"""Main application loop"""
		running = True

		print("Controls:")
		print("- Any key (except 'd'): Generate new pattern")
		print("- 'd': Toggle pose debug visualization")
		print("- ESC/Close: Exit")
		print(f"IMX500 status: {'Available' if self.imx500 else 'Not available'}")

		while running:
			running = self.handle_events()

			self.draw_pattern()

			if self.debug_mode:
				self.draw_pose_debug()

			self.draw_ui()
			self.update()

			pygame.display.flip()
			self.clock.tick(25)  # 25 FPS

		# Cleanup
		if self.picam2:
			self.picam2.stop()
		pygame.quit()

if __name__ == "__main__":
	try:
		app = MejkingApp()
		app.run()
	except KeyboardInterrupt:
		print("\nExiting...")
	except Exception as e:
		print(f"Error: {e}")
		print("Make sure you're running on a Raspberry Pi with IMX500 camera")
		print("Required model: /usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")