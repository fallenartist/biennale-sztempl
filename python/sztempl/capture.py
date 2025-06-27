# python/sztempl/capture.py

import sys
import time
import logging

import numpy as np
import pygame
import cv2
from picamera2 import Picamera2, CompletedRequest, MappedArray
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

from sztempl.config import settings

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

		# Camera init
		self.setup_imx500_camera()

		self.clock = pygame.time.Clock()

		print(f"Display: {self.display_width}x{self.display_height}, "
			  f"rotate90={self.rotate_90_degrees}, mirror={self.mirror_horizontally}")
		print("ESC: exit | d: debug | r: rotate toggle | m: mirror toggle")

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

	def draw_video_display(self):
		self.update_fps()
		self.screen.fill(self.black)
		if self.current_frame is None:
			font = pygame.font.Font(None, 74)
			t = font.render("Initializing camera...", True, self.white)
			r = t.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(t, r)
			return

		try:
			rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
			surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
			crop = self.get_face_shoulder_crop_region()
			if crop and self.pose_detected:
				x, y, w, h = crop
				if w > 0 and h > 0:
					cs = pygame.Surface((w, h))
					cs.blit(surf, (0, 0), (x, y, w, h))
					ds = pygame.transform.scale(
						cs, (self.display_width, self.display_height)
					)
				else:
					ds = self.scale_with_aspect_ratio(
						surf, self.display_width, self.display_height
					)
			else:
				ds = self.scale_with_aspect_ratio(
					surf, self.display_width, self.display_height
				)

			if self.mirror_horizontally:
				ds = pygame.transform.flip(ds, True, False)
			if self.rotate_90_degrees:
				ds = pygame.transform.rotate(ds, 90)
			self.screen.blit(ds, (0, 0))

			if self.debug_mode:
				self.draw_pose_debug_overlay()
			self.draw_status_overlay()

		except Exception as e:
			print(f"Video draw error: {e}", file=sys.stderr)
			font = pygame.font.Font(None, 48)
			t = font.render(f"Video error: {e}", True, self.red)
			r = t.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(t, r)

	def draw_pose_debug_overlay(self):
		if not self.pose_detected:
			return
		idx = np.argmax(self.last_scores)
		pts = self.last_keypoints[idx]
		inds = [0, 1, 2, 3, 4, 5, 6]
		crop = self.get_face_shoulder_crop_region()
		scr_pts = {}
		for j in inds:
			x, y, v = pts[j]
			if v > self.visibility_threshold:
				if crop:
					cx, cy, cw, ch = crop
					rx, ry = (x - cx) / cw, (y - cy) / ch
					sx, sy = rx * self.display_width, ry * self.display_height
				else:
					sx = x / self.camera_resolution[0] * self.display_width
					sy = y / self.camera_resolution[1] * self.display_height
				if self.mirror_horizontally:
					sx = self.display_width - sx
				if self.rotate_90_degrees:
					sx, sy = sy, self.display_width - sx
				scr_pts[j] = (int(sx), int(sy))

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
		font = pygame.font.Font(None, 48)
		fps_text = f"FPS: {self.current_fps:.1f}"
		if self.rotate_90_degrees:
			txt = pygame.transform.rotate(font.render(fps_text, True, self.white), 90)
			pos = (self.display_width - 20, 20)
		else:
			txt = font.render(fps_text, True, self.white)
			pos = (self.display_width - 200, 20)
		self.screen.blit(txt, pos)

		if hasattr(self, 'actual_camera_resolution'):
			res = f"CAM: {self.actual_camera_resolution[0]}x{self.actual_camera_resolution[1]}"
			if self.rotate_90_degrees:
				rt = pygame.transform.rotate(font.render(res, True, self.white), 90)
				rp = (self.display_width - 20, 20 + txt.get_width() + 10)
			else:
				rt = font.render(res, True, self.white)
				rp = (self.display_width - 300, 70)
			self.screen.blit(rt, rp)

	def handle_events(self):
		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				return False
			if e.type == pygame.KEYDOWN:
				if e.key == pygame.K_ESCAPE:
					return False
				if e.key == pygame.K_d:
					self.debug_mode = not self.debug_mode
					print("Debug mode", self.debug_mode)
				if e.key == pygame.K_r:
					self.rotate_90_degrees = not self.rotate_90_degrees
					if self.rotate_90_degrees:
						self.display_width, self.display_height = (
							self.fullhd_height, self.fullhd_width
						)
					else:
						self.display_width, self.display_height = (
							self.fullhd_width, self.fullhd_height
						)
					self.screen = pygame.display.set_mode(
						(self.display_width, self.display_height),
						pygame.FULLSCREEN
					)
					print(f"Rotation set to {90 if self.rotate_90_degrees else 0}°")
				if e.key == pygame.K_m:
					self.mirror_horizontally = not self.mirror_horizontally
					print("Mirror set to", self.mirror_horizontally)
		return True

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
