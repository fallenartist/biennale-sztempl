import os
import glob
import random
import bisect
import pygame
import cv2
import numpy as np
import math
import time

class PRNG:
	"""Pseudo-Random Number Generator with Perlin noise support - Python port of Processing version"""

	def __init__(self, seed=None):
		if seed is None:
			seed = int(time.time() * 1000) % 2147483647

		self.a = 1664525
		self.c = 1013904223
		self.m32 = 0xFFFFFFFF
		self.seed = seed % 2147483647

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

class TileFilter:
	def __init__(self, tiles_dir, cell_sizes,
				 noise_size_distribution=None,
				 enable_rotation=False,
				 enable_noise_random_rotation=False,
				 noise_rotation_interval=60,
				 enable_filter_random_rotation=False,
				 filter_rotation_interval=45):
		"""
		tiles_dir: path to directory containing tiles named '000.png', '012.png', etc.
		cell_sizes: list of ints, e.g. [40, 80, 120] for small/medium/large tiles.
		noise_size_distribution: dict with size weights for noise mode, e.g. {40: 1, 80: 3, 120: 2}
		enable_rotation: bool, whether to randomly rotate tiles
		enable_noise_random_rotation: bool, whether to randomly re-rotate tiles in noise mode
		noise_rotation_interval: int, frames between random rotation updates in noise mode
		enable_filter_random_rotation: bool, whether to randomly re-rotate tiles in filter mode
		filter_rotation_interval: int, frames between random rotation updates in filter mode
		"""
		self.cell_sizes = sorted(cell_sizes)
		self.enable_rotation = enable_rotation
		self.enable_noise_random_rotation = enable_noise_random_rotation
		self.noise_rotation_interval = noise_rotation_interval
		self.enable_filter_random_rotation = enable_filter_random_rotation
		self.filter_rotation_interval = filter_rotation_interval
		self.tiles_dir = tiles_dir

		# Store size distributions for both modes
		self.noise_size_distribution = noise_size_distribution
		self.filter_size_distribution = None  # Can be set later

		# Create weighted lists for size selection
		self._update_size_choices()

		# Force clear any potential caches on startup
		self._clear_caches()

		# discover and sort tile files by their filename brightness
		self._discover_tile_files()

		self.images = None  # will load once display is ready

		# Pattern generation variables (like Processing sketch)
		self.current_seed = int(random.random() * 1000000)
		self.prng = PRNG(self.current_seed)
		self.ii = list(range(self.num_images))  # image indices
		self.iish = []  # shuffled indices
		self.d0 = 0
		self.d1 = 0
		self.dz = 0
		self.dv = 0
		self.rr = []  # random positions for layer 2
		self.max_step = max(cell_sizes) if cell_sizes else 120
		self.min_step = min(cell_sizes) if cell_sizes else 40

		# Static rotation cache for detection mode
		self.detection_rotation_cache = {}
		self.detection_cache_valid = False

		# Slow tile size variation for detection mode
		self.detection_size_cache = {}
		self.detection_size_update_counter = 0
		self.detection_size_update_interval = 30  # Update sizes every 30 frames (slow)
		self.detection_size_update_probability = 0.02  # 2% chance per position per frame

		# Noise mode random rotation system
		self.noise_rotation_cache = {}
		self.noise_rotation_counter = 0

		# Filter mode random rotation system
		self.filter_rotation_cache = {}
		self.filter_rotation_counter = 0

		self.reset_pattern_parameters()

	def _update_size_choices(self):
		"""Update weighted size choice lists for both modes"""
		# Noise size choices
		if self.noise_size_distribution:
			self.noise_size_choices = []
			for size, weight in self.noise_size_distribution.items():
				if size in self.cell_sizes:
					self.noise_size_choices.extend([size] * weight)
		else:
			self.noise_size_choices = self.cell_sizes  # Default: equal weights

		# Filter size choices
		if self.filter_size_distribution:
			self.filter_size_choices = []
			for size, weight in self.filter_size_distribution.items():
				if size in self.cell_sizes:
					self.filter_size_choices.extend([size] * weight)
		else:
			self.filter_size_choices = self.cell_sizes  # Default: equal weights

	def _clear_caches(self):
		"""Clear various caches that might interfere with file reloading"""
		import gc

		# Force garbage collection
		gc.collect()

		# Clear pygame's internal caches if any
		try:
			import pygame.image
			# pygame doesn't have an explicit cache clear, but this helps
			pygame.image.get_extended()
		except:
			pass

		print("Cleared caches on startup")

	def _discover_tile_files(self):
		"""Discover and catalog tile files"""
		paths = glob.glob(os.path.join(self.tiles_dir, "*.png"))

		if not paths:
			print(f"WARNING: No PNG files found in {self.tiles_dir}")
			print("Looking for files like: 000.png, 001.png, etc.")

		levels = []
		for p in paths:
			name = os.path.splitext(os.path.basename(p))[0]
			try:
				b = int(name)
				levels.append((b, p))
				print(f"Found tile: {name}.png (brightness level {b})")
			except ValueError:
				print(f"Skipping non-numeric file: {os.path.basename(p)}")
				continue

		if not levels:
			raise RuntimeError(f"No valid tile PNGs found in {self.tiles_dir}")

		levels.sort(key=lambda x: x[0])
		self.brightness_levels = [b for b, _ in levels]
		self.paths = [p for _, p in levels]
		self.num_images = len(self.paths)

		print(f"Loaded {self.num_images} tile files:")
		for i, (brightness, path) in enumerate(levels):
			print(f"  {i}: {os.path.basename(path)} (level {brightness})")
		print()

	def _load_images(self):
		"""Load and convert all tile images after pygame.init/display."""
		print("Loading tile images into memory...")
		self.images = []

		for i, path in enumerate(self.paths):
			try:
				# Force fresh load by not using any cache
				img = pygame.image.load(path)
				img = img.convert_alpha()
				self.images.append(img)
				print(f"  Loaded: {os.path.basename(path)}")
			except Exception as e:
				print(f"  ERROR loading {path}: {e}")
				# Create a fallback colored square
				fallback = pygame.Surface((64, 64))
				fallback.fill((128, 128, 128))
				self.images.append(fallback)

		print(f"Successfully loaded {len(self.images)} tile images")

	def reload_images(self):
		"""Force reload of all tile images (clears cache)"""
		print("=== RELOADING TILE IMAGES ===")

		# Clear caches
		self._clear_caches()

		# Rediscover files
		self._discover_tile_files()

		# Force reload images
		self.images = None
		self._load_images()

		print("=== RELOAD COMPLETE ===\n")

	def reset_pattern_parameters(self):
		"""Reset pattern parameters with new random seed (like keyPressed in Processing)"""
		self.current_seed = int(random.random() * 1000000)

		# Use Python's random for initial setup, then switch to PRNG
		random.seed(self.current_seed)
		self.prng = PRNG(self.current_seed)

		# Initialize pattern parameters (matching Processing sketch exactly)
		self.d0 = random.uniform(100, 200)
		self.d1 = random.uniform(25, 75)
		self.dz = random.uniform(0, 100)
		self.dv = random.uniform(0, 0.05)  # Slower animation like Processing

		# Shuffle image indices
		self.ii = list(range(self.num_images))
		random.shuffle(self.ii)
		self.iish = self.ii.copy()

		# Invalidate detection cache when pattern changes
		self.detection_cache_valid = False
		self.detection_rotation_cache = {}
		self.detection_size_cache = {}
		self.detection_size_update_counter = 0

		# Reset noise rotation cache when pattern changes
		self.noise_rotation_cache = {}
		self.noise_rotation_counter = 0

		# Reset filter rotation cache when pattern changes
		self.filter_rotation_cache = {}
		self.filter_rotation_counter = 0

		print(f"New pattern generated with seed: {self.current_seed}")

	def update_pattern(self):
		"""Update animation parameters (like the dz += dv in Processing)"""
		self.dz += self.dv

	def update_noise_frame(self):
		"""Update noise mode frame counter for random rotation updates"""
		self.noise_rotation_counter += 1

	def update_detection_frame(self):
		"""Update detection mode frame counter for slow size changes"""
		self.detection_size_update_counter += 1

	def update_filter_frame(self):
		"""Update filter mode frame counter for random rotation updates"""
		self.filter_rotation_counter += 1

	def _get_tile_size_for_position(self, i, j, screen_w, screen_h, use_filter_distribution=False):
		"""Get tile size for a position with smooth random variation"""
		# Choose size distribution based on mode
		if use_filter_distribution and self.filter_size_distribution:
			size_choices = self.filter_size_choices
		elif not use_filter_distribution and self.noise_size_distribution:
			size_choices = self.noise_size_choices
		else:
			size_choices = self.cell_sizes  # Default

		# Create a grid key for this position (using base grid)
		base_step = self.min_step
		grid_x = i // base_step
		grid_y = j // base_step
		position_key = f"{grid_x}_{grid_y}"

		# Initialize with random size if not in cache
		if position_key not in self.detection_size_cache:
			self.detection_size_cache[position_key] = random.choice(size_choices)

		# Randomly update this position's size (smooth distributed changes)
		if random.random() < self.detection_size_update_probability:
			self.detection_size_cache[position_key] = random.choice(size_choices)

		return self.detection_size_cache[position_key]

	def _get_rotated_tile(self, img, size, rotation_key=None, is_noise_mode=False):
		"""Get a potentially rotated tile"""
		img_s = pygame.transform.scale(img, (size, size))

		if not self.enable_rotation:
			return img_s

		if is_noise_mode:
			if self.enable_noise_random_rotation:
				# Random rotation that updates every N frames
				if (rotation_key not in self.noise_rotation_cache or
					self.noise_rotation_counter % self.noise_rotation_interval == 0):
					# Randomly decide if this tile should rotate
					if random.random() < 0.3:  # 30% chance to get a new rotation
						self.noise_rotation_cache[rotation_key] = random.randint(0, 3) * 90
					elif rotation_key not in self.noise_rotation_cache:
						# Initialize with random rotation if not in cache
						self.noise_rotation_cache[rotation_key] = random.randint(0, 3) * 90

				angle = self.noise_rotation_cache.get(rotation_key, 0)
			else:
				# Static PRNG-based rotation (original behavior)
				angle = self.prng.next_int(4) * 90  # 0, 90, 180, 270 degrees
		else:
			# Filter/detection mode
			if self.enable_filter_random_rotation:
				# Random rotation that updates every N frames
				if (rotation_key not in self.filter_rotation_cache or
					self.filter_rotation_counter % self.filter_rotation_interval == 0):
					# Randomly decide if this tile should rotate
					if random.random() < 0.25:  # 25% chance (slightly less than noise)
						self.filter_rotation_cache[rotation_key] = random.randint(0, 3) * 90
					elif rotation_key not in self.filter_rotation_cache:
						# Initialize with random rotation if not in cache
						self.filter_rotation_cache[rotation_key] = random.randint(0, 3) * 90

				angle = self.filter_rotation_cache.get(rotation_key, 0)
			else:
				# Static cached rotation (original behavior)
				if rotation_key not in self.detection_rotation_cache:
					self.detection_rotation_cache[rotation_key] = random.randint(0, 3) * 90
				angle = self.detection_rotation_cache[rotation_key]

		if angle > 0:
			img_s = pygame.transform.rotate(img_s, angle)

		return img_s

	def apply_noise(self, screen):
		"""Generate pattern using PRNG noise (no face detected) - exactly like Processing sketch"""
		if self.images is None:
			self._load_images()

		screen_w, screen_h = screen.get_size()

		# Update animation (slow, mesmerizing flow)
		self.update_pattern()

		# Update noise frame counter for random rotation
		self.update_noise_frame()

		# Calculate random positions for layer 2 if not initialized
		if not self.rr:
			grid_points = (screen_w // self.max_step) * (screen_h // self.max_step)
			num_random_points = grid_points // random.randint(8, 13) * 2
			self.rr = []
			for i in range(0, num_random_points, 2):
				self.rr.append(random.randint(0, screen_w // self.max_step - 1))
				self.rr.append(random.randint(0, screen_h // self.max_step - 1))

		# Draw pattern layers EXACTLY like Processing (creates islands/patches)
		for p in range(len(self.cell_sizes)):
			step = self.cell_sizes[p]

			for j in range(0, screen_h, step):
				for i in range(0, screen_w, step):

					# Generate noise value (exactly matching Processing)
					n0 = self.prng.perlin(i / (self.d0 + self.dz), j / (self.d0 + self.dz))
					n1 = self.prng.perlin(i / (self.d1 + self.dz), j / (self.d1 + self.dz))
					n = 1 - (n0 * 0.75 + n1 * 0.25)

					# Convert to image index
					temp_k = max(0, min(int(n * self.num_images), self.num_images - 1))
					k = self.iish[temp_k] if temp_k < len(self.iish) else 0

					# Determine if we should draw (exactly matching Processing layer logic)
					should_draw = False

					if p == 0:  # Layer 0: draw all tiles
						should_draw = True
					elif p == 1 and k < 4:  # Layer 1: draw some shades only
						should_draw = True
					elif p == 2:  # Layer 2: draw at specific random positions
						for r in range(0, len(self.rr), 2):
							if (i == self.rr[r] * step and
								j == self.rr[r + 1] * step and
								self.rr[r] * step < screen_w and
								self.rr[r + 1] * step < screen_h):
								should_draw = True
								break

					if should_draw and k < len(self.images):
						# Draw tile with potential rotation
						img = self.images[k]
						rotation_key = f"noise_{i}_{j}_{step}_{k}"
						img_s = self._get_rotated_tile(img, step, rotation_key, is_noise_mode=True)
						screen.blit(img_s, (i, j))

	def apply_tiles(self, screen, frame):
		"""
		Apply tile pattern based on frame brightness (face detected)
		Linear brightness mapping with smooth tile size variation - FULL SCREEN COVERAGE
		"""
		if self.images is None:
			self._load_images()

		# Update frame counter for statistics
		self.update_detection_frame()

		# Update filter frame counter for random rotation
		self.update_filter_frame()

		# Convert capture to grayscale (NO INVERSION - direct linear mapping)
		if len(frame.shape) == 3:
			if frame.shape[2] == 3:
				gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			else:
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			gray = frame

		h, w = gray.shape
		screen_w, screen_h = screen.get_size()

		# Initialize rotation cache for this frame if needed
		if not self.detection_cache_valid:
			self.detection_rotation_cache = {}
			self.detection_cache_valid = True

		# FULL SCREEN COVERAGE: Use smallest tile size as base to ensure no gaps
		base_step = self.min_step

		# Track which areas have been filled by larger tiles
		filled_areas = set()

		# First pass: draw larger tiles and mark filled areas
		for step in reversed(self.cell_sizes):  # Start with largest tiles
			for j in range(0, screen_h, base_step):
				for i in range(0, screen_w, base_step):

					# Check if this area is already filled by a larger tile
					area_key = (i // base_step, j // base_step)
					if area_key in filled_areas:
						continue

					# Get tile size for this position (changes smoothly and randomly)
					chosen_size = self._get_tile_size_for_position(i, j, screen_w, screen_h, use_filter_distribution=True)

					# Only draw if the chosen size matches current step and position aligns
					if chosen_size == step and i % step == 0 and j % step == 0:

						# Sample brightness from face - DIRECT LINEAR MAPPING
						sample_x = max(0, min(i + step // 2, screen_w - 1))
						sample_y = max(0, min(j + step // 2, screen_h - 1))

						# Map screen coordinates to frame coordinates
						frame_x = int(sample_x * w / screen_w)
						frame_y = int(sample_y * h / screen_h)
						frame_x = max(0, min(frame_x, w - 1))
						frame_y = max(0, min(frame_y, h - 1))

						# Get brightness and map linearly: 0=darkest tile, 255=brightest tile
						try:
							brightness = gray[frame_y, frame_x]  # 0-255
							# Direct linear mapping: 0-255 → 0-(num_images-1)
							# 0 brightness → index 0 (000.png), 255 brightness → highest index
							idx = min(int(brightness * self.num_images / 256), self.num_images - 1)
						except (IndexError, ValueError):
							idx = 0  # Fallback to darkest tile

						# Bounds check
						if idx < len(self.images):
							# Draw tile with static rotation
							img = self.images[idx]
							rotation_key = f"detect_{i}_{j}_{step}_{idx}"
							img_s = self._get_rotated_tile(img, step, rotation_key, is_noise_mode=False)
							screen.blit(img_s, (i, j))

							# Mark this area and overlapping areas as filled
							for dy in range(0, step, base_step):
								for dx in range(0, step, base_step):
									if i + dx < screen_w and j + dy < screen_h:
										filled_key = ((i + dx) // base_step, (j + dy) // base_step)
										filled_areas.add(filled_key)

		# Second pass: fill any remaining gaps with smallest tiles
		for j in range(0, screen_h, base_step):
			for i in range(0, screen_w, base_step):
				area_key = (i // base_step, j // base_step)
				if area_key not in filled_areas:

					# Sample brightness for gap-filling tile
					sample_x = max(0, min(i + base_step // 2, screen_w - 1))
					sample_y = max(0, min(j + base_step // 2, screen_h - 1))

					frame_x = int(sample_x * w / screen_w)
					frame_y = int(sample_y * h / screen_h)
					frame_x = max(0, min(frame_x, w - 1))
					frame_y = max(0, min(frame_y, h - 1))

					try:
						brightness = gray[frame_y, frame_x]
						idx = min(int(brightness * self.num_images / 256), self.num_images - 1)
					except (IndexError, ValueError):
						idx = 0

					if idx < len(self.images):
						img = self.images[idx]
						rotation_key = f"detect_fill_{i}_{j}_{idx}"
						img_s = self._get_rotated_tile(img, base_step, rotation_key, is_noise_mode=False)
						screen.blit(img_s, (i, j))

	def generate_new_pattern(self):
		"""Public method to generate new pattern (like pressing a key in Processing)"""
		self.reset_pattern_parameters()
		self.rr = []  # Clear random positions to force regeneration

	def set_rotation_enabled(self, enabled):
		"""Enable or disable tile rotation"""
		self.enable_rotation = enabled
		if not enabled:
			self.detection_rotation_cache = {}  # Clear cache when disabling
			self.noise_rotation_cache = {}
			self.filter_rotation_cache = {}
		print(f"Tile rotation: {'enabled' if enabled else 'disabled'}")

	def set_noise_random_rotation(self, enabled, interval_frames=60):
		"""
		Enable/disable random rotation updates in noise mode
		enabled: bool, whether to randomly re-rotate tiles
		interval_frames: int, frames between rotation update cycles
		"""
		self.enable_noise_random_rotation = enabled
		self.noise_rotation_interval = interval_frames
		if not enabled:
			self.noise_rotation_cache = {}  # Clear cache when disabling
		print(f"Noise random rotation: {'enabled' if enabled else 'disabled'} "
			  f"(interval: {interval_frames} frames)")

	def set_filter_random_rotation(self, enabled, interval_frames=45):
		"""
		Enable/disable random rotation updates in filter mode
		enabled: bool, whether to randomly re-rotate tiles
		interval_frames: int, frames between rotation update cycles
		"""
		self.enable_filter_random_rotation = enabled
		self.filter_rotation_interval = interval_frames
		if not enabled:
			self.filter_rotation_cache = {}  # Clear cache when disabling
		print(f"Filter random rotation: {'enabled' if enabled else 'disabled'} "
			  f"(interval: {interval_frames} frames)")

	def set_noise_size_distribution(self, distribution):
		"""
		Set the size distribution for noise mode
		distribution: dict like {40: 1, 80: 4, 120: 3} (size: weight)
		"""
		self.noise_size_distribution = distribution
		self._update_size_choices()
		print(f"Noise size distribution updated: {distribution}")

	def set_filter_size_distribution(self, distribution):
		"""
		Set the size distribution for filter mode
		distribution: dict like {40: 2, 80: 1, 120: 1} (size: weight)
		"""
		self.filter_size_distribution = distribution
		self._update_size_choices()
		print(f"Filter size distribution updated: {distribution}")