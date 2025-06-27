import os
import glob
import random
import bisect
import pygame
import cv2
import numpy as np

class TileFilter:
	def __init__(self, tiles_dir, cell_sizes):
		"""
		tiles_dir: path to directory containing tiles named '000.png', '012.png', etc.
		cell_sizes: list of ints, e.g. [20, 40, 80] for small/medium/large tiles.
		"""
		self.cell_sizes = sorted(cell_sizes)
		# discover and sort tile files by their filename brightness
		paths = glob.glob(os.path.join(tiles_dir, "*.png"))
		levels = []
		for p in paths:
			name = os.path.splitext(os.path.basename(p))[0]
			try:
				b = int(name)
				levels.append((b, p))
			except ValueError:
				continue
		if not levels:
			raise RuntimeError(f"No tile PNGs found in {tiles_dir}")
		levels.sort(key=lambda x: x[0])
		self.brightness_levels = [b for b, _ in levels]
		self.paths = [p for _, p in levels]
		self.images = None  # will load once display is ready

	def _load_images(self):
		"""Load and convert all tile images after pygame.init/display."""
		self.images = [pygame.image.load(p).convert_alpha() for p in self.paths]

	def apply_noise(self, screen):
		"""Fill screen with random‐size tiles in a grid pattern."""
		if self.images is None:
			self._load_images()

		sw, sh = screen.get_size()
		base = self.cell_sizes[0]
		cols = sw // base + 1
		rows = sh // base + 1

		for y in range(rows):
			for x in range(cols):
				size = random.choice(self.cell_sizes)
				img = random.choice(self.images)
				img_s = pygame.transform.scale(img, (size, size))
				screen.blit(img_s, (x * base, y * base))

	def apply_tiles(self, screen, frame):
		"""
		Map each grid‐cell’s average brightness to a tile.
		Computes the mean brightness of the entire cell region
		and linearly maps 0–255 → 0–(n-1).
		"""
		if self.images is None:
			self._load_images()

		# 1) Ensure we have a color-to-gray conversion
		if frame.ndim == 3 and frame.shape[2] == 3:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		else:
			gray = frame.copy()

		h, w = gray.shape
		screen_w, screen_h = screen.get_size()
		base = self.cell_sizes[0]
		cols = max(screen_w // base, 1)
		rows = max(screen_h // base, 1)

		n = len(self.images)

		# For each cell, compute region average and map to tile
		for j in range(rows):
			y0 = int(j     * h / rows)
			y1 = int((j+1) * h / rows)
			for i in range(cols):
				x0 = int(i     * w / cols)
				x1 = int((i+1) * w / cols)

				# clamp
				y1c = min(max(y1, 0), h)
				x1c = min(max(x1, 0), w)
				cell = gray[y0:y1c, x0:x1c]

				if cell.size == 0:
					avg_b = 0
				else:
					avg_b = int(cell.mean())

				# map brightness → tile index
				idx = int(avg_b * n / 255)
				if idx >= n:
					idx = n - 1

				# draw
				img = self.images[idx]
				img_s = pygame.transform.scale(img, (base, base))
				screen.blit(img_s, (i * base, j * base))
