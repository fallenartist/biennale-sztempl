#!/usr/bin/env python3
"""
Simplified Face & Shoulders Detection (no rotations)
Based on biennale_draft.py - removed PRNG/tiles, focused on pose detection
Shows cropped face/shoulders video stream at full HD, always horizontal
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

		# Display settings (always horizontal)
		self.fullhd_width = 1920
		self.fullhd_height = 1080
		self.rotate_90_degrees = False   # no rotation
		self.mirror_horizontally = True  # Set to True to mirror image horizontally

		# Display dimensions
		self.display_width = self.fullhd_width
		self.display_height = self.fullhd_height

		# Pose detection thresholds
		self.visibility_threshold = 0.3      # Minimum keypoint visibility
		self.detection_threshold = 0.3       # Minimum person confidence

		# Crop margins
		self.crop_padding_factor = 0.2      # 20% padding around detected area

		# Camera settings
		self.camera_resolution = (2028, 1520)  # (W, H)
		self.pose_model_path = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

		# Model input size in (H, W)
		self.WINDOW_SIZE_H_W = (self.camera_resolution[1], self.camera_resolution[0])

		# FPS tracking
		self.frame_times = []
		self.frame_count = 0
		self.current_fps = 0.0

		# Colors
		self.black = (0, 0, 0)
		self.white = (255, 255, 255)
		self.green = (0, 255, 0)
		self.red = (255, 0, 0)

		# Initialize Pygame
		pygame.init()
		self.screen = pygame.display.set_mode((self.display_width, self.display_height), pygame.FULLSCREEN)
		pygame.display.set_caption("Face & Shoulders Detection - Full HD (No Rotation)")

		# Globals for pose data
		self.last_keypoints = None
		self.last_scores = None
		self.last_boxes = None
		self.pose_detected = False
		self.current_frame = None

		# Initialize camera + model
		self.setup_imx500_camera()

		self.clock = pygame.time.Clock()
		self.debug_mode = False

		print(f"Display: {self.display_width}x{self.display_height}, Rotation disabled, Mirror={'on' if self.mirror_horizontally else 'off'}")

	def setup_imx500_camera(self):
		"""Initialize IMX500 camera with pose estimation model"""
		try:
			self.imx500 = IMX500(self.pose_model_path)
			self.intrinsics = self.imx500.network_intrinsics
			if not self.intrinsics:
				self.intrinsics = NetworkIntrinsics()
				self.intrinsics.task = "pose estimation"
			elif self.intrinsics.task != "pose estimation":
				print("Warning: network not set for pose estimation", file=sys.stderr)

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

			print("IMX500 initialized:", config['main']['size'], "@", self.intrinsics.inference_rate, "fps")
			self.actual_camera_resolution = config['main']['size']
			self.WINDOW_SIZE_H_W = (self.actual_camera_resolution[1], self.actual_camera_resolution[0])

		except Exception as e:
			print(f"IMX500 init failed: {e}", file=sys.stderr)
			print("Falling back to standard camera")
			self.picam2 = Picamera2()
			config = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"}
			)
			self.picam2.configure(config)
			self.picam2.start()
			self.imx500 = None
			self.actual_camera_resolution = self.camera_resolution
			self.WINDOW_SIZE_H_W = (self.camera_resolution[1], self.camera_resolution[0])

	def update_fps(self):
		"""Update FPS calculation"""
		now = time.time()
		self.frame_times.append(now)
		self.frame_count += 1
		if len(self.frame_times) > 30:
			self.frame_times.pop(0)
		if self.frame_count % 10 == 0 and len(self.frame_times) > 1:
			span = self.frame_times[-1] - self.frame_times[0]
			self.current_fps = (len(self.frame_times)-1) / span if span>0 else 0.0

	def ai_output_tensor_parse(self, request: CompletedRequest):
		"""Grab frame & parse pose tensor"""
		try:
			with MappedArray(request, "main") as m:
				self.current_frame = m.array.copy()
		except:
			pass

		if not self.imx500:
			return

		try:
			outputs = self.imx500.get_outputs(metadata=request.get_metadata(), add_batch=True)
			if outputs is not None:
				keypoints, scores, boxes = postprocess_higherhrnet(
					outputs=outputs,
					img_size=self.WINDOW_SIZE_H_W,
					img_w_pad=(0,0),
					img_h_pad=(0,0),
					detection_threshold=self.detection_threshold,
					network_postprocess=True
				)
				if scores is not None and len(scores)>0:
					self.last_keypoints = np.reshape(np.stack(keypoints,axis=0),(len(scores),17,3))
					self.last_boxes = [np.array(b) for b in boxes]
					self.last_scores = scores
					self.pose_detected = True
					self.current_frame_crop_region = self._calculate_face_shoulder_crop_region()
				else:
					self.pose_detected = False
					self.current_frame_crop_region = None
			else:
				self.pose_detected = False
				self.current_frame_crop_region = None

		except Exception as e:
			print(f"Pose parse error: {e}", file=sys.stderr)
			self.pose_detected = False
			self.current_frame_crop_region = None

	def _calculate_face_shoulder_crop_region(self):
		"""Calculate crop region focusing on face & shoulders"""
		if not self.pose_detected or self.last_keypoints is None or len(self.last_scores)==0:
			return None

		idx = np.argmax(self.last_scores)
		pts = self.last_keypoints[idx]
		target = [0,1,2,3,4,5,6]  # nose, eyes, ears, shoulders
		valid = [(x,y) for (x,y,v) in pts[target] if v>self.visibility_threshold]
		if len(valid)<2:
			return None

		xs = [p[0] for p in valid]
		ys = [p[1] for p in valid]
		minx, maxx = min(xs), max(xs)
		miny, maxy = min(ys), max(ys)
		w, h = maxx-minx, maxy-miny
		padx, pady = w*self.crop_padding_factor, h*self.crop_padding_factor

		cx = max(0, minx-padx)
		cy = max(0, miny-pady)
		cw = min(self.camera_resolution[0]-cx, w+2*padx)
		ch = min(self.camera_resolution[1]-cy, h+2*pady)

		disp_aspect = self.display_width/self.display_height
		crop_aspect = cw/ch

		if crop_aspect>disp_aspect:
			new_h = cw/disp_aspect
			dy = (new_h-ch)/2
			cy = max(0, cy-dy)
			ch = min(self.camera_resolution[1]-cy, new_h)
		else:
			new_w = ch*disp_aspect
			dx = (new_w-cw)/2
			cx = max(0, cx-dx)
			cw = min(self.camera_resolution[0]-cx, new_w)

		return (int(cx), int(cy), int(cw), int(ch))

	def get_face_shoulder_crop_region(self):
		return getattr(self, 'current_frame_crop_region', None)

	def scale_with_aspect_ratio(self, surface, tw, th):
		sw, sh = surface.get_size()
		if sw/sh > tw/th:
			nw, nh = tw, int(tw * sh/sw)
		else:
			nh, nw = th, int(th * sw/sh)
		scaled = pygame.transform.scale(surface, (nw, nh))
		final = pygame.Surface((tw, th))
		final.fill(self.black)
		final.blit(scaled, ((tw-nw)//2, (th-nh)//2))
		return final

	def draw_video_display(self):
		self.update_fps()
		self.screen.fill(self.black)
		if self.current_frame is None:
			font = pygame.font.Font(None, 74)
			txt = font.render("Initializing camera...", True, self.white)
			rc = txt.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(txt, rc)
			return

		try:
			frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
			frame_surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0,1))
			crop = self.get_face_shoulder_crop_region()

			if crop and self.pose_detected:
				x,y,w,h = crop
				if w>0 and h>0:
					cs = pygame.Surface((w,h))
					cs.blit(frame_surf, (0,0), (x,y,w,h))
					display_surf = pygame.transform.scale(cs, (self.display_width, self.display_height))
				else:
					display_surf = self.scale_with_aspect_ratio(frame_surf, self.display_width, self.display_height)
			else:
				display_surf = self.scale_with_aspect_ratio(frame_surf, self.display_width, self.display_height)

			if self.mirror_horizontally:
				display_surf = pygame.transform.flip(display_surf, True, False)

			self.screen.blit(display_surf, (0,0))

			if self.debug_mode:
				self.draw_pose_debug_overlay()
			self.draw_status_overlay()

		except Exception as e:
			print(f"Video draw error: {e}", file=sys.stderr)
			font = pygame.font.Font(None, 48)
			txt = font.render(f"Video error: {e}", True, self.red)
			rc = txt.get_rect(center=(self.display_width//2, self.display_height//2))
			self.screen.blit(txt, rc)

	def draw_pose_debug_overlay(self):
		if not self.pose_detected or self.last_keypoints is None:
			return
		idx = np.argmax(self.last_scores)
		pts = self.last_keypoints[idx]
		names = ['nose','L_eye','R_eye','L_ear','R_ear','L_sh','R_sh']
		inds = [0,1,2,3,4,5,6]
		crop = self.get_face_shoulder_crop_region()

		screen_pts = {}
		for i, j in enumerate(inds):
			x,y,v = pts[j]
			if v>self.visibility_threshold:
				if crop:
					cx,cy,cw,ch = crop
					rx, ry = (x-cx)/cw, (y-cy)/ch
					sx, sy = rx*self.display_width, ry*self.display_height
				else:
					sx = x/self.camera_resolution[0]*self.display_width
					sy = y/self.camera_resolution[1]*self.display_height
				if self.mirror_horizontally:
					sx = self.display_width - sx
				screen_pts[j] = (int(sx), int(sy))

		# draw lines
		for (a,b) in [(0,1),(0,2),(1,3),(2,4)]:  # face
			if a in screen_pts and b in screen_pts:
				pygame.draw.line(self.screen, self.white, screen_pts[a], screen_pts[b], 2)
		if 5 in screen_pts and 6 in screen_pts:
			pygame.draw.line(self.screen, self.green, screen_pts[5], screen_pts[6], 2)
		# draw points + labels
		for i,j in enumerate(inds):
			if j in screen_pts:
				col = self.red if j==0 else (self.white if j<5 else self.green)
				pygame.draw.circle(self.screen, col, screen_pts[j], 8)
				lbl = pygame.font.Font(None,24).render(names[i], True, col)
				self.screen.blit(lbl, (screen_pts[j][0]+12, screen_pts[j][1]-12))

	def draw_status_overlay(self):
		font = pygame.font.Font(None, 48)
		fps_txt = f"FPS: {self.current_fps:.1f}"
		fps_surf = font.render(fps_txt, True, self.white)
		self.screen.blit(fps_surf, (self.display_width-200, 20))

		if hasattr(self, 'actual_camera_resolution'):
			res_txt = f"CAM: {self.actual_camera_resolution[0]}x{self.actual_camera_resolution[1]}"
			res_surf = font.render(res_txt, True, self.white)
			self.screen.blit(res_surf, (self.display_width-300, 70))

	def handle_events(self):
		for e in pygame.event.get():
			if e.type == pygame.QUIT:
				return False
			if e.type == pygame.KEYDOWN:
				if e.key == pygame.K_ESCAPE:
					return False
				if e.key == pygame.K_d:
					self.debug_mode = not self.debug_mode
					print("Debug:", "ON" if self.debug_mode else "OFF")
				if e.key == pygame.K_m:
					self.mirror_horizontally = not self.mirror_horizontally
					print("Mirror:", "ON" if self.mirror_horizontally else "OFF")
		return True

	def run(self):
		print("Controls: ESC exit, d debug toggle, m mirror toggle")
		running = True
		while running:
			running = self.handle_events()
			self.draw_video_display()
			pygame.display.flip()
			self.clock.tick(30)
		try: self.picam2.stop()
		except: pass
		pygame.quit()
		print("App closed")

if __name__ == "__main__":
	try:
		app = FaceShoulderApp()
		app.run()
	except KeyboardInterrupt:
		print("\nExiting...")
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		print("Ensure Raspberry Pi + IMX500 camera and model installed")