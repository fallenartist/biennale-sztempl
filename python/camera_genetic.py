#!/usr/bin/env python3
"""
IMX500 Genetic Tile Matching Art
Uses pose detection with genetic algorithm to evolve geometric tile placements
that match detected people using your actual tile images with rotations
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

class TileLibrary:
	"""Manages the geometric tile images and their rotations"""

	def __init__(self, tiles_dir: str = "tiles"):
		self.tiles_dir = tiles_dir
		self.original_tiles = []
		self.rotated_tiles = []  # [tile_idx][rotation] = pygame.Surface
		self.num_tiles = 0
		self.tile_size = 20  # Default size, will be updated

		self.load_tiles()

	def load_tiles(self):
		"""Load all tile images and create rotations"""
		if not os.path.exists(self.tiles_dir):
			print(f"Warning: {self.tiles_dir} directory not found, creating placeholder tiles")
			self.create_placeholder_tiles()
			return

		# Load tiles F00.png through F07.png (matching your original project)
		for i in range(8):
			tile_path = os.path.join(self.tiles_dir, f"F{i:02d}.png")
			if os.path.exists(tile_path):
				try:
					tile_img = pygame.image.load(tile_path)
					self.original_tiles.append(tile_img)
					print(f"Loaded {tile_path}")
				except pygame.error as e:
					print(f"Error loading {tile_path}: {e}")
			else:
				print(f"Warning: {tile_path} not found")

		if not self.original_tiles:
			print("No tiles found, creating placeholder tiles")
			self.create_placeholder_tiles()
			return

		# Add programmatic empty and full tiles
		self.add_special_tiles()

		self.num_tiles = len(self.original_tiles)
		self.tile_size = self.original_tiles[0].get_width()  # Assume square tiles

		# Create rotations for each tile (0°, 90°, 180°, 270°)
		self.rotated_tiles = []
		for tile in self.original_tiles:
			rotations = []
			for angle in [0, 90, 180, 270]:
				rotated = pygame.transform.rotate(tile, angle)
				rotations.append(rotated)
			self.rotated_tiles.append(rotations)

		print(f"Loaded {self.num_tiles} tiles (including empty/full) with 4 rotations each")

	def create_placeholder_tiles(self):
		"""Create only empty and full tiles if no PNG files found"""
		self.tile_size = 20

		# Only create empty and full tiles, no geometric placeholders
		empty_tile = pygame.Surface((self.tile_size, self.tile_size))
		empty_tile.fill((255, 255, 255))  # All white

		full_tile = pygame.Surface((self.tile_size, self.tile_size))
		full_tile.fill((0, 0, 0))  # All black

		self.original_tiles = [empty_tile, full_tile]
		self.num_tiles = 2

		print("No PNG tiles found - using only empty and full tiles")

		# Create rotations
		self.rotated_tiles = []
		for tile in self.original_tiles:
			rotations = []
			for angle in [0, 90, 180, 270]:
				rotated = pygame.transform.rotate(tile, angle)
				rotations.append(rotated)
			self.rotated_tiles.append(rotations)

	def add_special_tiles(self):
		"""Add empty (all white) and full (all black) tiles programmatically"""
		if not self.original_tiles:
			return

		# Get tile size from existing tiles
		tile_size = self.original_tiles[0].get_width()

		# Create empty tile (all white)
		empty_tile = pygame.Surface((tile_size, tile_size))
		empty_tile.fill((255, 255, 255))  # All white
		self.original_tiles.append(empty_tile)

		# Create full tile (all black)
		full_tile = pygame.Surface((tile_size, tile_size))
		full_tile.fill((0, 0, 0))  # All black
		self.original_tiles.append(full_tile)

		print("Added empty (white) and full (black) tiles")

	def create_diamond(self):
		"""Create diamond shape"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))  # White background
		points = [
			(self.tile_size // 2, 2),
			(self.tile_size - 2, self.tile_size // 2),
			(self.tile_size // 2, self.tile_size - 2),
			(2, self.tile_size // 2)
		]
		pygame.draw.polygon(surface, (0, 0, 0), points)
		return surface

	def create_triangle_up(self):
		"""Create upward triangle"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))
		points = [
			(self.tile_size // 2, 2),
			(self.tile_size - 2, self.tile_size - 2),
			(2, self.tile_size - 2)
		]
		pygame.draw.polygon(surface, (0, 0, 0), points)
		return surface

	def create_triangle_down(self):
		"""Create downward triangle"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))
		points = [
			(2, 2),
			(self.tile_size - 2, 2),
			(self.tile_size // 2, self.tile_size - 2)
		]
		pygame.draw.polygon(surface, (0, 0, 0), points)
		return surface

	def create_triangle_left(self):
		"""Create left triangle"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))
		points = [
			(2, self.tile_size // 2),
			(self.tile_size - 2, 2),
			(self.tile_size - 2, self.tile_size - 2)
		]
		pygame.draw.polygon(surface, (0, 0, 0), points)
		return surface

	def create_triangle_right(self):
		"""Create right triangle"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))
		points = [
			(2, 2),
			(2, self.tile_size - 2),
			(self.tile_size - 2, self.tile_size // 2)
		]
		pygame.draw.polygon(surface, (0, 0, 0), points)
		return surface

	def create_cross(self):
		"""Create cross shape"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))
		thickness = 4
		center = self.tile_size // 2
		# Vertical line
		pygame.draw.rect(surface, (0, 0, 0),
						(center - thickness//2, 2, thickness, self.tile_size - 4))
		# Horizontal line
		pygame.draw.rect(surface, (0, 0, 0),
						(2, center - thickness//2, self.tile_size - 4, thickness))
		return surface

	def create_circle(self):
		"""Create circle shape"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))
		pygame.draw.circle(surface, (0, 0, 0),
						  (self.tile_size // 2, self.tile_size // 2),
						  self.tile_size // 2 - 2)
		return surface

	def create_square(self):
		"""Create square shape"""
		surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
		surface.fill((255, 255, 255))
		pygame.draw.rect(surface, (0, 0, 0),
						(2, 2, self.tile_size - 4, self.tile_size - 4))
		return surface

	def get_tile(self, tile_idx: int, rotation: int, size: int, accent_color: Optional[Tuple[int, int, int]] = None) -> pygame.Surface:
		"""Get a tile with specified rotation and size, optionally recolored"""
		if tile_idx >= self.num_tiles or rotation >= 4:
			tile_idx = tile_idx % self.num_tiles
			rotation = rotation % 4

		# Get base tile
		base_tile = self.rotated_tiles[tile_idx][rotation]

		# Scale if needed
		if size != self.tile_size:
			scaled_tile = pygame.transform.scale(base_tile, (size, size))
		else:
			scaled_tile = base_tile.copy()

		# Apply accent color if specified - ONLY recolor black shapes, keep white background
		if accent_color is not None:
			# Convert to pixel array for manipulation
			pixel_array = pygame.surfarray.array3d(scaled_tile)

			# Find black pixels (the shapes) - strict black detection
			black_mask = (pixel_array[:, :, 0] < 30) & (pixel_array[:, :, 1] < 30) & (pixel_array[:, :, 2] < 30)

			# Only replace the black shape pixels with accent color, leave white background alone
			if np.any(black_mask):
				pixel_array[black_mask] = accent_color

				# Create new surface with recolored shapes
				colored_tile = pygame.Surface((size, size))
				pygame.surfarray.blit_array(colored_tile, pixel_array)
				return colored_tile

		return scaled_tile

class TilePlacement:
	"""Represents a single tile placement in the genetic individual"""

	def __init__(self, x: int, y: int, size: int, tile_idx: int, rotation: int, use_accent: bool = False):
		self.x = x
		self.y = y
		self.size = size
		self.tile_idx = tile_idx
		self.rotation = rotation
		self.use_accent = use_accent
		self.fitness = 0.0

	def mutate(self, num_tiles: int, mutation_rate: float = 0.1):
		"""Mutate tile placement properties - closer to Anastasia Opara's approach"""
		if random.random() < mutation_rate:
			# More focused mutations - change tile type (most important for matching)
			self.tile_idx = random.randint(0, num_tiles - 1)

		if random.random() < mutation_rate * 0.3:  # Lower chance for rotation
			# Change rotation (less frequent than tile changes)
			self.rotation = random.randint(0, 3)

	def crossover(self, other: 'TilePlacement') -> 'TilePlacement':
		"""Create offspring from two parent tile placements"""
		new_x = (self.x + other.x) // 2
		new_y = (self.y + other.y) // 2
		new_size = random.choice([self.size, other.size])
		new_tile_idx = random.choice([self.tile_idx, other.tile_idx])
		new_rotation = random.choice([self.rotation, other.rotation])
		new_use_accent = random.choice([self.use_accent, other.use_accent])

		return TilePlacement(new_x, new_y, new_size, new_tile_idx, new_rotation, new_use_accent)

class ColorAnalyzer:
	"""Analyzes camera frames for dominant colors and their positions"""

	def __init__(self):
		self.dominant_color = (128, 128, 128)  # Default gray
		self.color_map = None

		# Predetermined bright colors to snap to
		self.bright_colors = [
			(255, 0, 0),    # Red
			(0, 255, 0),    # Green
			(0, 0, 255),    # Blue
			(255, 255, 0),  # Yellow
			(255, 0, 255),  # Magenta
			(0, 255, 255),  # Cyan
			(255, 128, 0),  # Orange
			(128, 0, 255),  # Purple
		]

	def analyze_frame(self, frame: np.ndarray) -> Tuple[Tuple[int, int, int], np.ndarray]:
		"""Analyze frame for dominant color and create position map"""
		# Resize for faster processing
		small_frame = cv2.resize(frame, (160, 120))

		# Find dominant non-grayscale color using simple histogram approach
		pixels = small_frame.reshape(-1, 3)

		# Filter out near-grayscale pixels and collect colorful ones
		color_pixels = []
		for pixel in pixels:
			r, g, b = pixel
			# Check if pixel has enough color saturation
			max_val = max(r, g, b)
			min_val = min(r, g, b)
			if max_val - min_val > 30:  # Has some color saturation
				color_pixels.append(pixel)

		if len(color_pixels) > 10:
			# Simple method: find most common color region
			color_pixels = np.array(color_pixels)

			# Quantize colors to reduce noise (divide by 32, then multiply back)
			quantized = (color_pixels // 32) * 32

			# Find unique colors and their counts
			unique_colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)

			# Get the most frequent color
			most_frequent_idx = np.argmax(counts)
			detected_color = tuple(map(int, unique_colors[most_frequent_idx]))

			# Snap to closest predetermined bright color
			self.dominant_color = self.snap_to_bright_color(detected_color)
		else:
			# Fallback to a default color
			self.dominant_color = (64, 128, 192)  # Blue from your original project

		# Create color position map
		self.create_color_map(small_frame)

		return self.dominant_color, self.color_map

	def snap_to_bright_color(self, detected_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
		"""Snap detected color to closest predetermined bright color"""
		min_distance = float('inf')
		closest_color = self.bright_colors[0]

		for bright_color in self.bright_colors:
			# Calculate Euclidean distance in RGB space
			distance = sum((a - b) ** 2 for a, b in zip(detected_color, bright_color)) ** 0.5
			if distance < min_distance:
				min_distance = distance
				closest_color = bright_color

		return closest_color

	def create_color_map(self, frame: np.ndarray):
		"""Create a map showing where the dominant color appears"""
		h, w = frame.shape[:2]
		self.color_map = np.zeros((h, w), dtype=np.uint8)

		# Calculate color distance threshold
		target_color = np.array(self.dominant_color)

		for y in range(h):
			for x in range(w):
				pixel_color = frame[y, x]
				# Calculate color distance
				distance = np.linalg.norm(pixel_color - target_color)

				# If pixel is close to dominant color, mark it
				if distance < 50:  # Threshold for color similarity
					self.color_map[y, x] = 255

	def posterize_frame(self, frame: np.ndarray) -> np.ndarray:
		"""Convert frame to black and white only (no accent color)"""
		# Convert to grayscale for thresholding
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

		# Create posterized version - only black and white
		posterized = np.zeros_like(frame)

		# Simple threshold - adjust this value to control black/white balance
		threshold = 128

		# Black regions
		dark_mask = gray < threshold
		posterized[dark_mask] = [0, 0, 0]      # Black

		# White regions
		light_mask = gray >= threshold
		posterized[light_mask] = [255, 255, 255]  # White

		return posterized

class GeneticTileMatcher:
	"""Genetic algorithm for matching tiles to target image"""

	def __init__(self, width: int, height: int, tile_library: TileLibrary, population_size: int = 40):
		self.width = width
		self.height = height
		self.tile_library = tile_library
		self.population_size = population_size
		self.population: List[List[TilePlacement]] = []
		self.target_image: Optional[np.ndarray] = None
		self.accent_color: Optional[Tuple[int, int, int]] = None
		self.generation = 0

		# Tile size configuration based on pose detection
		self.small_size = 16    # Fine detail areas
		self.medium_size = 32   # Medium detail areas
		self.large_size = 64    # Background areas

		# Genetic algorithm parameters - closer to original genetic drawing approach
		self.mutation_rate = 0.08  # Lower mutation rate for more gradual evolution
		self.crossover_rate = 0.6  # Moderate crossover rate
		self.elitism_count = 3     # Fewer elite individuals

		# Performance optimization
		self.fitness_cache = {}

	def create_tile_grid(self, person_keypoints: Optional[np.ndarray] = None) -> List[TilePlacement]:
		"""Create a grid of tile placements with size based on proximity to person features"""
		placements = []

		# Create influence map for tile sizing (same as before)
		influence_map = np.ones((self.height, self.width)) * self.large_size

		if person_keypoints is not None:
			# Face keypoints (nose, eyes, ears)
			face_indices = [0, 1, 2, 3, 4]
			for idx in face_indices:
				if idx < len(person_keypoints):
					x, y, confidence = person_keypoints[idx]
					if confidence > 0.3:
						cv2.circle(influence_map, (int(x), int(y)), 60, self.small_size, -1)

			# Shoulder/neck keypoints
			shoulder_indices = [5, 6]
			for idx in shoulder_indices:
				if idx < len(person_keypoints):
					x, y, confidence = person_keypoints[idx]
					if confidence > 0.3:
						cv2.circle(influence_map, (int(x), int(y)), 80, self.medium_size, -1)

		# Generate tile placements based on influence map
		step_size = self.small_size
		for y in range(0, self.height - step_size, step_size):
			for x in range(0, self.width - step_size, step_size):
				tile_size = int(influence_map[y, x])

				# Skip if this position should use a larger tile
				if tile_size > step_size:
					if (x % tile_size == 0) and (y % tile_size == 0):
						# Random tile selection - no accent color usage for now
						tile_idx = random.randint(0, self.tile_library.num_tiles - 1)
						rotation = random.randint(0, 3)
						use_accent = False  # Disable accent colors for now

						placements.append(TilePlacement(x, y, tile_size, tile_idx, rotation, use_accent))
				else:
					# Small tile - no accent color
					tile_idx = random.randint(0, self.tile_library.num_tiles - 1)
					rotation = random.randint(0, 3)
					use_accent = False  # Disable accent colors for now

					placements.append(TilePlacement(x, y, tile_size, tile_idx, rotation, use_accent))

		return placements

	def initialize_population(self, person_keypoints: Optional[np.ndarray] = None):
		"""Initialize population with random individuals"""
		self.population = []
		for _ in range(self.population_size):
			individual = self.create_tile_grid(person_keypoints)
			self.population.append(individual)
		self.generation = 0
		self.fitness_cache.clear()

	def render_individual(self, individual: List[TilePlacement]) -> pygame.Surface:
		"""Render an individual (list of tile placements) to a pygame surface"""
		surface = pygame.Surface((self.width, self.height))
		surface.fill((255, 255, 255))  # White background

		# Sort by size (largest first) for proper layering
		sorted_placements = sorted(individual, key=lambda p: p.size, reverse=True)

		for placement in sorted_placements:
			accent_color = self.accent_color if placement.use_accent else None
			tile_surface = self.tile_library.get_tile(
				placement.tile_idx,
				placement.rotation,
				placement.size,
				accent_color
			)
			surface.blit(tile_surface, (placement.x, placement.y))

		return surface

	def calculate_fitness(self, individual: List[TilePlacement]) -> float:
		"""Calculate fitness by comparing rendered individual to target image using tile-by-tile analysis"""
		if self.target_image is None:
			return 0.0

		# Create cache key
		cache_key = hash(tuple((p.x, p.y, p.size, p.tile_idx, p.rotation, p.use_accent) for p in individual))
		if cache_key in self.fitness_cache:
			return self.fitness_cache[cache_key]

		# Render individual
		rendered = self.render_individual(individual)
		rendered_array = pygame.surfarray.array3d(rendered)
		rendered_array = np.transpose(rendered_array, (1, 0, 2))

		# Resize target to match
		target_resized = cv2.resize(self.target_image, (self.width, self.height))

		# Convert both to grayscale for comparison
		if len(target_resized.shape) == 3:
			target_gray = cv2.cvtColor(target_resized, cv2.COLOR_RGB2GRAY)
		else:
			target_gray = target_resized

		rendered_gray = cv2.cvtColor(rendered_array, cv2.COLOR_RGB2GRAY)

		# Tile-by-tile fitness calculation for more granular comparison
		tile_fitness_sum = 0.0
		tile_count = 0

		for placement in individual:
			# Extract the region for this tile from both images
			x, y, size = placement.x, placement.y, placement.size

			# Ensure we don't go out of bounds
			if x + size <= self.width and y + size <= self.height:
				target_region = target_gray[y:y+size, x:x+size]
				rendered_region = rendered_gray[y:y+size, x:x+size]

				if target_region.size > 0 and rendered_region.size > 0:
					# Calculate MSE for this specific tile region
					mse = np.mean((target_region.astype(float) - rendered_region.astype(float)) ** 2)
					tile_fitness = 1.0 / (1.0 + mse / 100.0)  # More sensitive to local changes

					# Weight smaller tiles more heavily (they should be more accurate)
					weight = 1.0 + (64 - size) / 64.0  # Smaller tiles get higher weight

					tile_fitness_sum += tile_fitness * weight
					tile_count += weight

		# Average tile fitness
		if tile_count > 0:
			avg_tile_fitness = tile_fitness_sum / tile_count
		else:
			avg_tile_fitness = 0.0

		# Global fitness metrics (less weight than tile-by-tile)
		# Overall MSE
		mse_global = np.mean((target_gray.astype(float) - rendered_gray.astype(float)) ** 2)
		global_fitness = 1.0 / (1.0 + mse_global / 1000.0)

		# Edge similarity
		target_edges = cv2.Canny(target_gray, 50, 150)
		rendered_edges = cv2.Canny(rendered_gray, 50, 150)
		edge_similarity = np.sum(target_edges & rendered_edges) / (np.sum(target_edges | rendered_edges) + 1)

		# Combine metrics: heavily weight tile-by-tile comparison
		fitness = (avg_tile_fitness * 0.7 + global_fitness * 0.2 + edge_similarity * 0.1)

		# Cache result
		self.fitness_cache[cache_key] = fitness
		return fitness

	def tournament_selection(self, population: List[List[TilePlacement]],
						   fitness_scores: List[float], tournament_size: int = 3) -> List[TilePlacement]:
		"""Select individual using tournament selection"""
		tournament_indices = random.sample(range(len(population)),
										 min(tournament_size, len(population)))
		tournament_fitness = [fitness_scores[i] for i in tournament_indices]
		winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
		return population[winner_idx]

	def crossover_individuals(self, parent1: List[TilePlacement], parent2: List[TilePlacement]) -> List[TilePlacement]:
		"""Create offspring from two parents"""
		child = []
		min_len = min(len(parent1), len(parent2))

		for i in range(min_len):
			if random.random() < 0.5:
				child.append(parent1[i].crossover(parent2[i]))
			else:
				child.append(parent2[i].crossover(parent1[i]))

		return child

	def mutate_individual(self, individual: List[TilePlacement]):
		"""Mutate an individual - more conservative approach similar to Anastasia's genetic drawing"""
		# Mutate fewer tiles per generation but with higher probability per tile
		# This creates more gradual, focused evolution
		for placement in individual:
			placement.mutate(self.tile_library.num_tiles, self.mutation_rate)

	def evolve_generation(self):
		"""Evolve the population for one generation"""
		# Calculate fitness for all individuals
		fitness_scores = []
		for individual in self.population:
			fitness = self.calculate_fitness(individual)
			fitness_scores.append(fitness)

		# Sort population by fitness
		sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population),
												 key=lambda pair: pair[0], reverse=True)]

		new_population = []

		# Elitism - keep best individuals
		for i in range(self.elitism_count):
			new_population.append([p.__class__(p.x, p.y, p.size, p.tile_idx, p.rotation, p.use_accent) for p in sorted_population[i]])

		# Generate offspring
		while len(new_population) < self.population_size:
			# Selection
			parent1 = self.tournament_selection(sorted_population, fitness_scores)
			parent2 = self.tournament_selection(sorted_population, fitness_scores)

			# Crossover
			if random.random() < self.crossover_rate:
				child = self.crossover_individuals(parent1, parent2)
			else:
				child = [p.__class__(p.x, p.y, p.size, p.tile_idx, p.rotation, p.use_accent) for p in random.choice([parent1, parent2])]

			# Mutation
			self.mutate_individual(child)

			new_population.append(child)

		self.population = new_population
		self.generation += 1

		# Clear fitness cache periodically
		if self.generation % 15 == 0:
			self.fitness_cache.clear()

	def get_best_individual(self) -> List[TilePlacement]:
		"""Get the best individual from current population"""
		if not self.population:
			return []

		best_fitness = -1
		best_individual = self.population[0]

		for individual in self.population:
			fitness = self.calculate_fitness(individual)
			if fitness > best_fitness:
				best_fitness = fitness
				best_individual = individual

		return best_individual

	def set_target_image(self, image: np.ndarray, accent_color: Tuple[int, int, int]):
		"""Set the target image and accent color for evolution"""
		self.target_image = image
		self.accent_color = accent_color
		self.fitness_cache.clear()

class GeneticTileArtApp:
	"""Main application combining IMX500 pose detection with genetic tile matching"""

	def __init__(self):
		# Screen configuration
		self.width = 640
		self.height = 480

		# Initialize Pygame
		pygame.init()
		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption("IMX500 Genetic Tile Art - Mejking Style")

		# Colors
		self.black = (0, 0, 0)
		self.white = (255, 255, 255)
		self.green = (0, 255, 0)
		self.red = (255, 0, 0)
		self.blue = (0, 126, 192)  # Your original blue

		# Crop mode configuration
		self.crop_modes = ["FULL_POSE", "FACE_ONLY", "HEAD_SHOULDERS"]
		self.current_crop_mode = 0
		self.crop_margins = {
			"FULL_POSE": 30,        # Small margin around whole pose
			"FACE_ONLY": 40,        # Medium margin around face
			"HEAD_SHOULDERS": 35    # Medium margin around head and shoulders
		}

		# Load tile library
		self.tile_library = TileLibrary("tiles")

		# Color analysis
		self.color_analyzer = ColorAnalyzer()

		# Pose detection
		self.last_keypoints = None
		self.last_scores = None
		self.pose_detected = False
		self.detection_threshold = 0.3
		self.WINDOW_SIZE_H_W = (480, 640)

		# Genetic tile matching
		self.genetic_matcher = GeneticTileMatcher(self.width, self.height, self.tile_library, population_size=25)
		self.evolution_active = False
		self.generations_per_frame = 1

		# Camera frame storage
		self.current_frame = None
		self.frozen_crop = None          # Static crop for comparison
		self.frozen_posterized = None    # Static posterized target
		self.current_accent_color = self.blue

		# Setup camera
		self.setup_imx500_camera()

		# UI
		self.font = pygame.font.Font(None, 20)
		self.clock = pygame.time.Clock()

		print("Controls:")
		print("- SPACE/C: Freeze frame and start evolution")
		print("- R: Reset population")
		print("- S: Save best individual as image")
		print("- M: Cycle crop mode (Full/Face/Head+Shoulders)")
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
				self.intrinsics.inference_rate = 10

			self.intrinsics.update_with_defaults()

			# Initialize camera
			self.picam2 = Picamera2(self.imx500.camera_num)
			config = self.picam2.create_preview_configuration(
				main={"size": (640, 480), "format": "RGB888"},
				controls={'FrameRate': self.intrinsics.inference_rate},
				buffer_count=12
			)

			self.picam2.configure(config)
			self.picam2.pre_callback = self.ai_output_tensor_parse

			self.imx500.show_network_fw_progress_bar()
			self.picam2.start(config, show_preview=False)
			self.imx500.set_auto_aspect_ratio()

			print("IMX500 pose estimation initialized successfully")

		except Exception as e:
			print(f"Failed to initialize IMX500: {e}")
			# Fallback to regular camera
			self.picam2 = Picamera2()
			config = self.picam2.create_preview_configuration(
				main={"size": (640, 480), "format": "RGB888"}
			)
			self.picam2.configure(config)
			self.picam2.start()
			self.imx500 = None

	def ai_output_tensor_parse(self, request: CompletedRequest):
		"""Parse pose estimation output"""
		if not self.imx500:
			return

		try:
			# Capture current frame
			with MappedArray(request, "main") as m:
				self.current_frame = m.array.copy()

			# Get pose outputs
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

					# Analyze frame for colors and posterize
					self.process_frame()
				else:
					self.pose_detected = False
			else:
				self.pose_detected = False

		except Exception as e:
			print(f"Error parsing pose results: {e}")
			self.pose_detected = False

	def extract_crop_region(self) -> Optional[np.ndarray]:
		"""Extract crop region based on current crop mode"""
		if not self.pose_detected or self.last_keypoints is None or self.current_frame is None:
			return None

		try:
			# Find best person (highest confidence)
			best_person_idx = np.argmax(self.last_scores)
			person_keypoints = self.last_keypoints[best_person_idx]

			crop_mode = self.crop_modes[self.current_crop_mode]
			margin = self.crop_margins[crop_mode]

			if crop_mode == "FACE_ONLY":
				# Face keypoints only
				keypoint_indices = [0, 1, 2, 3, 4]  # nose, eyes, ears
			elif crop_mode == "HEAD_SHOULDERS":
				# Face + shoulders
				keypoint_indices = [0, 1, 2, 3, 4, 5, 6]  # face + shoulders
			else:  # FULL_POSE
				# All visible keypoints
				keypoint_indices = list(range(17))

			# Collect valid points
			valid_points = []
			for idx in keypoint_indices:
				if idx < len(person_keypoints):
					x, y, confidence = person_keypoints[idx]
					if confidence > self.detection_threshold:
						valid_points.append((int(x), int(y)))

			if len(valid_points) >= 2:
				# Calculate bounding box
				x_coords = [p[0] for p in valid_points]
				y_coords = [p[1] for p in valid_points]

				min_x, max_x = min(x_coords), max(x_coords)
				min_y, max_y = min(y_coords), max(y_coords)

				# Add margin
				crop_x1 = max(0, min_x - margin)
				crop_y1 = max(0, min_y - margin)
				crop_x2 = min(self.current_frame.shape[1], max_x + margin)
				crop_y2 = min(self.current_frame.shape[0], max_y + margin)

				# Extract crop
				crop = self.current_frame[crop_y1:crop_y2, crop_x1:crop_x2]
				return crop if crop.size > 0 else None

		except Exception as e:
			print(f"Error extracting crop: {e}")

		return None

	def process_frame(self):
		"""Process current frame for color analysis - don't update frozen target"""
		if self.current_frame is None:
			return

		try:
			# Extract crop region based on current crop mode for color analysis
			crop_region = self.extract_crop_region()

			if crop_region is not None:
				# Analyze colors from the crop region (detect but don't use accent color)
				accent_color, color_map = self.color_analyzer.analyze_frame(crop_region)
				self.current_accent_color = accent_color  # Save for display but don't use in tiles

		except Exception as e:
			print(f"Error processing frame: {e}")

	def capture_target_frame(self):
		"""Capture and freeze current frame as static target for genetic evolution"""
		crop_region = self.extract_crop_region()

		if crop_region is not None and self.pose_detected:
			# Freeze the crop and posterize it
			self.frozen_crop = crop_region.copy()
			self.frozen_posterized = self.color_analyzer.posterize_frame(crop_region)

			# Set target for genetic algorithm (black and white only)
			self.genetic_matcher.set_target_image(self.frozen_posterized, None)  # No accent color

			# Initialize population with current pose
			best_person_idx = np.argmax(self.last_scores)
			person_keypoints = self.last_keypoints[best_person_idx]
			self.genetic_matcher.initialize_population(person_keypoints)

			crop_mode_name = self.crop_modes[self.current_crop_mode]
			print(f"Frame frozen! Mode: {crop_mode_name}, Detected accent: {self.current_accent_color} (saved but not used)")
			return True
		else:
			print("No valid frame or pose detected")
			return False

	def handle_events(self):
		"""Handle pygame events"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					return False
				elif event.key == pygame.K_SPACE or event.key == pygame.K_c:
					# Freeze frame and start evolution (combined C and SPACE functionality)
					if self.pose_detected:
						if self.capture_target_frame():
							self.evolution_active = True
							print("Frame frozen and evolution started!")
						else:
							print("Failed to capture target frame")
					else:
						print("No pose detected - cannot freeze frame")
				elif event.key == pygame.K_m:
					# Cycle crop mode
					old_mode = self.crop_modes[self.current_crop_mode]
					self.current_crop_mode = (self.current_crop_mode + 1) % len(self.crop_modes)
					new_mode = self.crop_modes[self.current_crop_mode]
					print(f"Crop mode: {old_mode} -> {new_mode}")

					# If currently evolving, reset population with new crop
					if self.evolution_active:
						self.evolution_active = False
						if self.capture_target_frame():
							self.evolution_active = True
							print("Population reset with new crop mode!")
				elif event.key == pygame.K_r:
					# Reset population
					if self.pose_detected and self.posterized_frame is not None:
						best_person_idx = np.argmax(self.last_scores)
						person_keypoints = self.last_keypoints[best_person_idx]
						self.genetic_matcher.initialize_population(person_keypoints)
						print("Population reset")
					else:
						print("No pose detected for reset")
				elif event.key == pygame.K_s:
					# Save best individual
					if self.genetic_matcher.population:
						best = self.genetic_matcher.get_best_individual()
						surface = self.genetic_matcher.render_individual(best)
						filename = f"genetic_tile_art_gen_{self.genetic_matcher.generation}.png"
						pygame.image.save(surface, filename)
						print(f"Saved {filename}")
		return True

	def draw_frame(self):
		"""Draw current frame"""
		self.screen.fill(self.black)

		# Layout: Camera preview (top-left), Target (bottom-left), Genetic art (right)
		camera_rect = pygame.Rect(10, 10, 200, 150)
		target_rect = pygame.Rect(10, 170, 200, 150)
		art_rect = pygame.Rect(220, 10, 410, 310)
		info_rect = pygame.Rect(10, 330, 620, 140)

		# Draw camera frame
		if self.current_frame is not None:
			frame_surface = pygame.surfarray.make_surface(
				np.transpose(self.current_frame, (1, 0, 2))
			)
			scaled_frame = pygame.transform.scale(frame_surface, camera_rect.size)
			self.screen.blit(scaled_frame, camera_rect.topleft)

			# Draw pose keypoints on camera frame
			if self.pose_detected and self.last_keypoints is not None:
				for person_idx in range(len(self.last_scores)):
					person_keypoints = self.last_keypoints[person_idx]
					for kp_idx, (x, y, confidence) in enumerate(person_keypoints):
						if confidence > self.detection_threshold:
							screen_x = int(x * camera_rect.width / 640) + camera_rect.left
							screen_y = int(y * camera_rect.height / 480) + camera_rect.top
							pygame.draw.circle(self.screen, self.green, (screen_x, screen_y), 2)

		# Draw camera frame border and label
		pygame.draw.rect(self.screen, self.white, camera_rect, 2)
		camera_label = self.font.render("Camera Feed", True, self.white)
		self.screen.blit(camera_label, (camera_rect.left, camera_rect.top - 20))

		# Draw target (frozen posterized) frame
		if self.frozen_posterized is not None:
			target_surface = pygame.surfarray.make_surface(
				np.transpose(self.frozen_posterized, (1, 0, 2))
			)
			scaled_target = pygame.transform.scale(target_surface, target_rect.size)
			self.screen.blit(scaled_target, target_rect.topleft)

		# Draw target frame border and label
		pygame.draw.rect(self.screen, self.white, target_rect, 2)
		target_label = self.font.render("Frozen Target (B&W)", True, self.white)
		self.screen.blit(target_label, (target_rect.left, target_rect.top - 20))

		# Draw genetic art
		if self.genetic_matcher.population and self.evolution_active:
			best_individual = self.genetic_matcher.get_best_individual()
			art_surface = self.genetic_matcher.render_individual(best_individual)
			scaled_art = pygame.transform.scale(art_surface, art_rect.size)
			self.screen.blit(scaled_art, art_rect.topleft)
		elif self.genetic_matcher.population:
			# Show paused state
			best_individual = self.genetic_matcher.get_best_individual()
			art_surface = self.genetic_matcher.render_individual(best_individual)
			scaled_art = pygame.transform.scale(art_surface, art_rect.size)
			self.screen.blit(scaled_art, art_rect.topleft)
		else:
			# Show placeholder
			pygame.draw.rect(self.screen, (50, 50, 50), art_rect)
			placeholder_text = self.font.render("Press C to capture target", True, self.white)
			text_rect = placeholder_text.get_rect(center=art_rect.center)
			self.screen.blit(placeholder_text, text_rect)

		# Draw genetic art border and label
		pygame.draw.rect(self.screen, self.white, art_rect, 2)
		art_label = self.font.render("Genetic Tile Evolution", True, self.white)
		self.screen.blit(art_label, (art_rect.left, art_rect.top - 20))

		# Draw info panel
		pygame.draw.rect(self.screen, (30, 30, 30), info_rect)
		pygame.draw.rect(self.screen, self.white, info_rect, 1)

		# Info text
		y_offset = info_rect.top + 10
		line_height = 18

		# Status line
		evolution_status = "EVOLVING" if self.evolution_active else "PAUSED"
		evolution_color = self.green if self.evolution_active else self.red
		pose_status = "POSE DETECTED" if self.pose_detected else "NO POSE"
		pose_color = self.green if self.pose_detected else self.red

		status_text = f"Evolution: "
		text_surface = self.font.render(status_text, True, self.white)
		self.screen.blit(text_surface, (info_rect.left + 10, y_offset))

		status_surface = self.font.render(evolution_status, True, evolution_color)
		self.screen.blit(status_surface, (info_rect.left + 10 + text_surface.get_width(), y_offset))

		pose_text = f" | Pose: "
		pose_text_surface = self.font.render(pose_text, True, self.white)
		self.screen.blit(pose_text_surface, (info_rect.left + 10 + text_surface.get_width() + status_surface.get_width(), y_offset))

		pose_status_surface = self.font.render(pose_status, True, pose_color)
		self.screen.blit(pose_status_surface, (info_rect.left + 10 + text_surface.get_width() + status_surface.get_width() + pose_text_surface.get_width(), y_offset))

		y_offset += line_height

		# Generation and fitness
		if self.genetic_matcher.population:
			gen_text = f"Generation: {self.genetic_matcher.generation}"
			gen_surface = self.font.render(gen_text, True, self.white)
			self.screen.blit(gen_surface, (info_rect.left + 10, y_offset))

			best_fitness = self.genetic_matcher.calculate_fitness(self.genetic_matcher.get_best_individual())
			fitness_text = f"Best Fitness: {best_fitness:.4f}"
			fitness_surface = self.font.render(fitness_text, True, self.white)
			self.screen.blit(fitness_surface, (info_rect.left + 200, y_offset))

			y_offset += line_height

		# Accent color display (detected but not used)
		accent_text = f"Detected Color: RGB{self.current_accent_color} (not used)"
		accent_surface = self.font.render(accent_text, True, self.white)
		self.screen.blit(accent_surface, (info_rect.left + 10, y_offset))

		# Draw color swatch
		color_rect = pygame.Rect(info_rect.left + 200, y_offset, 30, 15)
		pygame.draw.rect(self.screen, self.current_accent_color, color_rect)
		pygame.draw.rect(self.screen, self.white, color_rect, 1)

		y_offset += line_height

		# Tile library info
		tile_text = f"Loaded {self.tile_library.num_tiles} tiles, {self.tile_library.num_tiles * 4} variants (with rotations)"
		tile_surface = self.font.render(tile_text, True, self.white)
		self.screen.blit(tile_surface, (info_rect.left + 10, y_offset))

		y_offset += line_height

		# Crop mode display
		crop_mode_name = self.crop_modes[self.current_crop_mode]
		crop_text = f"Crop Mode: {crop_mode_name}"
		crop_surface = self.font.render(crop_text, True, self.white)
		self.screen.blit(crop_surface, (info_rect.left + 10, y_offset))

		y_offset += line_height

		# Controls
		controls_text = "SPACE/C: Freeze & Evolve | M: Crop Mode | R: Reset | S: Save | ESC: Exit"
		controls_surface = self.font.render(controls_text, True, (200, 200, 200))
		self.screen.blit(controls_surface, (info_rect.left + 10, y_offset))

	def run(self):
		"""Main application loop"""
		running = True

		print(f"Mejking 2021 Genetic Tile Art Started")
		print(f"Tile library: {self.tile_library.num_tiles} tiles loaded")

		while running:
			running = self.handle_events()

			# Evolution step
			if self.evolution_active and self.genetic_matcher.population:
				for _ in range(self.generations_per_frame):
					self.genetic_matcher.evolve_generation()

			self.draw_frame()
			pygame.display.flip()
			self.clock.tick(25)  # 25 FPS (matching your original project)

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
		print("Place your geometric tile images (F00.png - F07.png) in the 'tiles' directory")