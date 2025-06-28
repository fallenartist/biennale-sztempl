#!/usr/bin/env python3
"""
Genetic Mutation Module for Tile Filter System - SCREEN SIZE FIXED

The genetic mutation output MUST cover the full screen exactly like the tile filter does.
"""

import numpy as np
import cv2
import pygame
import random
import math
import time
import os
from typing import Optional, Tuple, List
import glob

# GPIO imports (graceful fallback if not available)
try:
	import RPi.GPIO as GPIO
	GPIO_AVAILABLE = True
except ImportError:
	GPIO_AVAILABLE = False

class GeneticTileEvolution:
	"""
	Genetic algorithm for evolving tile mosaics to match target images
	"""

	def __init__(self, tile_library, target_size: Tuple[int, int]):
		self.tile_library = tile_library
		self.target_size = target_size  # (width, height) - MUST BE SCREEN SIZE
		self.scales = [40, 80, 120]  # Match tile filter sizes
		self.scale_thresholds = [80, 180]  # Detail mask brightness thresholds

		# Genetic algorithm state
		self.regions = []  # List of (row, col, scale) for each tile
		self.genome = []   # List of tile variant indices
		self.mosaic = None # Current rendered mosaic
		self.target_image = None # B&W target to match
		self.detail_mask = None  # Mask determining tile sizes
		self.best_score = float('inf')

		# Evolution parameters
		self.max_iterations = 3000
		self.current_iteration = 0
		self.is_evolving = False

	def create_gradient_detail_mask(self) -> np.ndarray:
		"""Create detail mask using gradient-based sampling (pure Sobel, no keypoints)"""
		mask = np.zeros(self.target_size[::-1], dtype=np.uint8)  # Height x Width

		if self.target_image is not None:
			# Convert target to grayscale for gradient calculation
			if len(self.target_image.shape) == 3:
				gray = cv2.cvtColor(self.target_image, cv2.COLOR_RGB2GRAY)
			else:
				gray = self.target_image.copy()

			# Calculate gradient using Sobel operators
			gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
			gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)

			# Calculate gradient magnitude
			mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

			# Normalize magnitudes to 0-1 range
			if mag.max() > 0:
				mag = mag / mag.max()

			# Lower contrast to make transitions smoother
			mag = np.power(mag, 0.3)

			# Apply gaussian blur for smoother transitions
			blur_size = max(21, self.target_size[0] // 30)
			if blur_size % 2 == 0:
				blur_size += 1
			mag = cv2.GaussianBlur(mag, (blur_size, blur_size), 0)

			# Convert to 0-255 range for mask
			mask = (mag * 255).astype(np.uint8)

		return mask

	def setup_regions(self, pose_keypoints: Optional[np.ndarray],
					 camera_resolution: Tuple[int, int]):
		"""Setup grid regions based on detail mask"""
		self.detail_mask = self.create_gradient_detail_mask()
		self.regions = []

		# Process each tile scale
		for scale in self.scales:
			grid_h = math.ceil(self.target_size[1] / scale)
			grid_w = math.ceil(self.target_size[0] / scale)

			for i in range(grid_h):
				for j in range(grid_w):
					y1, x1 = i * scale, j * scale
					y2 = min(y1 + scale, self.target_size[1])
					x2 = min(x1 + scale, self.target_size[0])

					# Calculate average detail mask value
					mask_region = self.detail_mask[y1:y2, x1:x2]
					avg_detail = mask_region.mean()

					# Choose scale based on detail level
					if avg_detail >= self.scale_thresholds[1]:
						chosen_scale = self.scales[2]  # 120px (lowest detail)
					elif avg_detail >= self.scale_thresholds[0]:
						chosen_scale = self.scales[1]  # 80px (medium detail)
					else:
						chosen_scale = self.scales[0]  # 40px (highest detail)

					# Only add this region if it matches current scale
					if chosen_scale == scale:
						self.regions.append((i, j, scale))

		# Initialize random genome
		self.genome = [random.randrange(self.tile_library.num_variants) for _ in self.regions]

	def render_mosaic(self) -> np.ndarray:
		"""Render current genome to mosaic image - EXACT SCREEN SIZE"""
		# Create mosaic with EXACT target size
		mosaic = np.ones((self.target_size[1], self.target_size[0]), dtype=np.uint8) * 255

		for (i, j, scale), variant_idx in zip(self.regions, self.genome):
			y1, x1 = i * scale, j * scale
			y2 = min(y1 + scale, self.target_size[1])
			x2 = min(x1 + scale, self.target_size[0])

			tile = self.tile_library.render_variant(variant_idx, scale)

			# Handle edge cases
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

	def initialize(self, target_image: np.ndarray, pose_keypoints: Optional[np.ndarray],
				  camera_resolution: Tuple[int, int]):
		"""Initialize genetic algorithm with target image and pose"""
		self.target_image = target_image
		self.setup_regions(pose_keypoints, camera_resolution)
		self.mosaic = self.render_mosaic()
		self.best_score = self.calculate_score()
		self.current_iteration = 0
		self.is_evolving = True

	def evolve_step(self) -> bool:
		"""Perform one evolution step"""
		if not self.regions or self.current_iteration >= self.max_iterations:
			self.is_evolving = False
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

			# Calculate local error
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

	def get_progress(self) -> float:
		"""Get evolution progress as percentage (0.0 to 1.0)"""
		if self.max_iterations == 0:
			return 1.0
		return min(1.0, self.current_iteration / self.max_iterations)

	def reset(self):
		"""Reset evolution state"""
		self.regions = []
		self.genome = []
		self.mosaic = None
		self.target_image = None
		self.detail_mask = None
		self.best_score = float('inf')
		self.current_iteration = 0
		self.is_evolving = False

class SimpleTileLibrary:
	"""Simple tile library that works with existing tile filter images"""

	def __init__(self, tiles_dir: str):
		self.tiles_dir = tiles_dir
		self.tile_variants = []
		self.num_variants = 0
		self.base_tile_size = 40

		self.load_tiles()

	def load_tiles(self):
		"""Load PNG files, convert to B&W, create rotations"""
		tile_paths = glob.glob(os.path.join(self.tiles_dir, "*.png"))

		if not tile_paths:
			self.create_basic_tiles()
			return

		base_tiles = []

		# Load all PNG files and convert to B&W
		for path in sorted(tile_paths):
			if "eye.png" in path:  # Skip special eye tile
				continue

			try:
				img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
				if img is None:
					continue

				# Handle different channel formats
				if img.ndim == 2:  # Grayscale
					b = g = r = img
					a = np.full_like(img, 255)
					img = cv2.merge([b, g, r, a])
				elif img.shape[2] == 3:  # RGB
					b, g, r = cv2.split(img)
					a = np.full_like(b, 255)
					img = cv2.merge([b, g, r, a])

				# Composite onto white background
				rgb = img[:, :, :3].astype(float)
				alpha = img[:, :, 3:4].astype(float) / 255.0
				composited = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)

				# Convert to B&W
				gray = cv2.cvtColor(composited, cv2.COLOR_BGR2GRAY)
				_, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

				base_tiles.append(bw)

			except Exception as e:
				pass

		if not base_tiles:
			self.create_basic_tiles()
			return

		# Set tile size from first loaded tile
		self.base_tile_size = base_tiles[0].shape[0]

		# Add empty and full tiles
		empty_tile = np.ones((self.base_tile_size, self.base_tile_size), dtype=np.uint8) * 255
		full_tile = np.zeros((self.base_tile_size, self.base_tile_size), dtype=np.uint8)
		base_tiles.extend([empty_tile, full_tile])

		# Create rotation variants
		self.tile_variants = []
		for tile in base_tiles:
			for angle in [0, 90, 180, 270]:
				center = (tile.shape[1] / 2, tile.shape[0] / 2)
				M = cv2.getRotationMatrix2D(center, angle, 1)
				rotated = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]),
									   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
				self.tile_variants.append(rotated)

		self.num_variants = len(self.tile_variants)

	def create_basic_tiles(self):
		"""Create basic geometric tiles if no PNGs found"""
		self.base_tile_size = 40
		size = self.base_tile_size

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

		# Empty and full
		empty = np.ones((size, size), dtype=np.uint8) * 255
		full = np.zeros((size, size), dtype=np.uint8)
		shapes.extend([empty, full])

		# Create rotations
		self.tile_variants = []
		for tile in shapes:
			for angle in [0, 90, 180, 270]:
				center = (tile.shape[1] / 2, tile.shape[0] / 2)
				M = cv2.getRotationMatrix2D(center, angle, 1)
				rotated = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]),
									   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
				self.tile_variants.append(rotated)

		self.num_variants = len(self.tile_variants)

	def render_variant(self, variant_idx: int, target_size: int) -> np.ndarray:
		"""Render a specific tile variant at the requested size"""
		if variant_idx >= self.num_variants:
			variant_idx = variant_idx % self.num_variants

		tile = self.tile_variants[variant_idx]
		if target_size != self.base_tile_size:
			return cv2.resize(tile, (target_size, target_size), interpolation=cv2.INTER_AREA)
		return tile.copy()

class GeneticMutationModule:
	"""Main genetic mutation module"""

	def __init__(self, tiles_dir: str, gpio_pin: int = 18):
		self.tiles_dir = tiles_dir
		self.gpio_pin = gpio_pin

		# GPIO setup for capacitive sensor
		self.gpio_enabled = False
		if GPIO_AVAILABLE:
			try:
				GPIO.setmode(GPIO.BCM)
				GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
				self.gpio_enabled = True
			except Exception as e:
				pass

		# Load tile library for genetic algorithm
		self.tile_library = SimpleTileLibrary(tiles_dir)

		# Evolution state
		self.genetic_evolution = None
		self.is_active = False
		self.frozen_frame = None
		self.steps_per_frame = 3

		# Progress bar settings
		self.progress_bar_width = 8
		self.progress_bar_color = (0, 100, 255)  # Blue

		# Toggle detection variables (only for GPIO now)
		self.last_gpio_state = False
		self.gpio_press_cooldown = 0
		self.cooldown_frames = 10

	def check_triggers(self, keys_pressed) -> bool:
		"""DEPRECATED: Use check_gpio_trigger() for GPIO and handle SPACE in main event loop"""
		return False

	def check_gpio_trigger(self) -> bool:
		"""Check for GPIO sensor trigger only"""
		should_toggle = False

		# Decrease GPIO cooldown
		if self.gpio_press_cooldown > 0:
			self.gpio_press_cooldown -= 1

		# Check GPIO sensor (detect signal change)
		if self.gpio_enabled:
			try:
				current_gpio_state = GPIO.input(self.gpio_pin)
				if current_gpio_state and not self.last_gpio_state and self.gpio_press_cooldown == 0:
					should_toggle = True
					self.gpio_press_cooldown = self.cooldown_frames
					self._handle_gpio_toggle()
				self.last_gpio_state = current_gpio_state
			except Exception as e:
				pass

		return should_toggle

	def _handle_gpio_toggle(self):
		"""Handle GPIO toggle - needs to be called from main app context"""
		pass

	def toggle_mode(self, frame: np.ndarray = None, pose_keypoints: Optional[np.ndarray] = None,
				   camera_resolution: Tuple[int, int] = None, target_size: Tuple[int, int] = None):
		"""Toggle between genetic mode and normal tile filter mode"""
		if not self.is_active:
			# Start genetic mode
			if frame is not None and camera_resolution is not None and target_size is not None:
				success = self.start_evolution(frame, pose_keypoints, camera_resolution, target_size)
		else:
			# Return to normal mode
			self.reset()

	def start_evolution(self, frame: np.ndarray, pose_keypoints: Optional[np.ndarray],
					   camera_resolution: Tuple[int, int], target_size: Tuple[int, int]):
		"""Start genetic evolution with frozen frame"""
		if frame is None:
			return False

		try:
			# Store frozen frame
			self.frozen_frame = frame.copy()

			# Make sure the target image is EXACTLY the target size
			if frame.shape[:2] != (target_size[1], target_size[0]):
				# Resize frame to EXACT target size
				frame = cv2.resize(frame, target_size)

			# Convert to grayscale first
			if len(frame.shape) == 3:
				gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			else:
				gray = frame

			# Threshold to B&W
			_, bw_frame = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

			# Convert back to RGB for processing
			target_image = cv2.cvtColor(bw_frame, cv2.COLOR_GRAY2RGB)

			# Initialize genetic evolution with EXACT target size
			self.genetic_evolution = GeneticTileEvolution(self.tile_library, target_size)
			self.genetic_evolution.initialize(target_image, pose_keypoints, camera_resolution)

			self.is_active = True
			return True

		except Exception as e:
			return False

	def update(self):
		"""Update genetic evolution (call each frame)"""
		if not self.is_active or not self.genetic_evolution:
			return

		# Run evolution steps
		for _ in range(self.steps_per_frame):
			if not self.genetic_evolution.evolve_step():
				break

	def draw_progress_bar(self, screen):
		"""Draw blue progress bar on right edge of screen (full to empty, top to bottom)"""
		if not self.is_active or not self.genetic_evolution:
			return

		screen_w, screen_h = screen.get_size()
		progress = self.genetic_evolution.get_progress()

		# Progress bar on right edge (full to empty, top to bottom)
		bar_x = screen_w - self.progress_bar_width
		bar_height = int(screen_h * (1.0 - progress))  # Full to empty as progress increases

		# Draw progress bar (blue line shrinking downward)
		if bar_height > 0:
			progress_rect = pygame.Rect(bar_x, 0, self.progress_bar_width, bar_height)
			pygame.draw.rect(screen, self.progress_bar_color, progress_rect)

	def draw_mosaic(self, screen, display_rect):
		"""Draw evolving mosaic over the display area - FORCE EXACT SCREEN COVERAGE"""
		if not self.is_active or not self.genetic_evolution or self.genetic_evolution.mosaic is None:
			return

		try:
			# Get the mosaic
			mosaic = self.genetic_evolution.mosaic

			# Convert mosaic to RGB for pygame (mosaic is grayscale)
			mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2RGB)

			# Convert to pygame surface - CRITICAL: Use correct axis ordering
			mosaic_surface = pygame.surfarray.make_surface(mosaic_rgb.swapaxes(0, 1))

			# Scale to EXACT display rect size - NO MATTER WHAT
			scaled_mosaic = pygame.transform.scale(mosaic_surface,
												 (display_rect.width, display_rect.height))

			# Draw over existing content at EXACT position
			screen.blit(scaled_mosaic, (display_rect.x, display_rect.y))

		except Exception as e:
			pass

	def draw_debug_mask(self, screen, display_rect):
		"""Draw the detail mask overlay at 80% opacity in debug mode"""
		if (not self.is_active or not self.genetic_evolution or
			self.genetic_evolution.detail_mask is None):
			return

		try:
			# Convert detail mask to pygame surface
			mask = self.genetic_evolution.detail_mask

			# Create colored mask (white = high detail, black = low detail)
			colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
			colored_mask[:,:,0] = mask  # Red channel
			colored_mask[:,:,1] = mask  # Green channel
			colored_mask[:,:,2] = mask  # Blue channel

			# Convert to pygame surface
			mask_surface = pygame.surfarray.make_surface(colored_mask.swapaxes(0, 1))

			# Scale to EXACT display rect size
			scaled_mask = pygame.transform.scale(mask_surface,
												(display_rect.width, display_rect.height))

			# Set alpha for 80% opacity
			scaled_mask.set_alpha(int(255 * 0.8))

			# Draw over existing content at EXACT position
			screen.blit(scaled_mask, (display_rect.x, display_rect.y))

		except Exception as e:
			pass

	def reset(self):
		"""Reset genetic mutation and return to normal operation"""
		self.is_active = False
		self.frozen_frame = None
		if self.genetic_evolution:
			self.genetic_evolution.reset()
			self.genetic_evolution = None

	def cleanup(self):
		"""Cleanup GPIO resources"""
		if self.gpio_enabled and GPIO_AVAILABLE:
			try:
				GPIO.cleanup()
			except:
				pass

def register(app_context):
	"""Module registration function"""
	pass