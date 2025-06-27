#!/usr/bin/env python3
"""
Face & Shoulders Detection
Margins, fit-scaling (contain), rotation, mirroring, pose overlay,
single corner log, FPS. Controls: r=rotate, m=mirror, d=debug, ESC=exit.
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
		# Screen & margins
		self.fullhd_width, self.fullhd_height = 1920, 1080
		self.margin_left, self.margin_top = 8, 8
		self.margin_right, self.margin_bottom = 88, 8
		self.inner_width = self.fullhd_width - self.margin_left - self.margin_right
		self.inner_height = self.fullhd_height - self.margin_top - self.margin_bottom

		# Controls
		self.rotate_90_degrees = True
		self.mirror_horizontally = True
		self.debug_mode = False
		self.logged_corners = False

		# Pose thresholds
		self.visibility_threshold = 0.3
		self.detection_threshold = 0.3
		self.crop_padding_factor = 0.2

		# Camera
		self.camera_resolution = (2028, 1520)
		self.pose_model_path = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

		# Stats
		self.current_fps = 0.0
		self.frame_times = []
		self.frame_count = 0

		# Colors
		self.black = (0, 0, 0)
		self.white = (255, 255, 255)
		self.green = (0, 255, 0)
		self.red = (255, 0, 0)

		# Init Pygame
		pygame.init()
		self.screen = pygame.display.set_mode((self.fullhd_width, self.fullhd_height), pygame.FULLSCREEN)
		pygame.display.set_caption("Face & Shoulders Detection")
		self.clock = pygame.time.Clock()

		# Pose data
		self.last_keypoints = None
		self.last_scores = None
		self.pose_detected = False
		self.current_frame = None
		self.crop_region = None

		# Camera setup
		self.setup_imx500_camera()

	def setup_imx500_camera(self):
		try:
			self.imx500 = IMX500(self.pose_model_path)
			intr = self.imx500.network_intrinsics or NetworkIntrinsics()
			intr.task = "pose estimation"
			intr.inference_rate = intr.inference_rate or 30
			intr.update_with_defaults()

			self.picam2 = Picamera2(self.imx500.camera_num)
			cfg = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"},
				controls={'FrameRate': intr.inference_rate}, buffer_count=12
			)
			self.picam2.pre_callback = self.ai_output_tensor_parse
			self.imx500.show_network_fw_progress_bar()
			self.picam2.start(cfg, show_preview=False)
			self.imx500.set_auto_aspect_ratio()
			self.actual_camera_resolution = cfg['main']['size']
		except Exception as e:
			print(f"Camera init failed: {e}", file=sys.stderr)
			self.picam2 = Picamera2()
			cfg = self.picam2.create_preview_configuration(
				main={"size": self.camera_resolution, "format": "RGB888"}
			)
			self.picam2.configure(cfg)
			self.picam2.start()
			self.imx500 = None
			self.actual_camera_resolution = self.camera_resolution

	def ai_output_tensor_parse(self, req: CompletedRequest):
		try:
			with MappedArray(req, 'main') as m:
				self.current_frame = m.array.copy()
		except:
			return
		if not self.imx500:
			return
		try:
			out = self.imx500.get_outputs(metadata=req.get_metadata(), add_batch=True)
			if out:
				kps, scores, _ = postprocess_higherhrnet(
					outputs=out,
					img_size=(self.actual_camera_resolution[1], self.actual_camera_resolution[0]),
					img_w_pad=(0,0), img_h_pad=(0,0),
					detection_threshold=self.detection_threshold,
					network_postprocess=True
				)
				if scores and len(scores) > 0:
					self.last_keypoints = np.reshape(np.stack(kps,axis=0),(len(scores),17,3))
					self.last_scores = scores
					self.pose_detected = True
					self.crop_region = self.calculate_crop()
				else:
					self.pose_detected = False
					self.crop_region = None
		except Exception as e:
			print(f"Pose parse error: {e}", file=sys.stderr)
			self.pose_detected = False
			self.crop_region = None

	def calculate_crop(self):
		if not self.pose_detected or self.last_keypoints is None:
			return None
		idx = np.argmax(self.last_scores)
		pts = self.last_keypoints[idx]
		inds = [0,1,2,3,4,5,6]
		valid = [(x,y) for (x,y,v) in pts[inds] if v > self.visibility_threshold]
		if len(valid) < 2:
			return None
		xs, ys = zip(*valid)
		minx, maxx = min(xs), max(xs)
		miny, maxy = min(ys), max(ys)
		w, h = maxx-minx, maxy-miny
		pad_w, pad_h = w*self.crop_padding_factor, h*self.crop_padding_factor
		cx, cy = max(0, minx-pad_w), max(0, miny-pad_h)
		fh, fw = self.current_frame.shape[:2]
		cw = min(fw-cx, w+2*pad_w)
		ch = min(fh-cy, h+2*pad_h)
		if cw <=0 or ch<=0:
			return None
		tgt_as = self.inner_width/self.inner_height
		crop_as = cw/ch
		if crop_as>tgt_as:
			new_h=cw/tgt_as; diff=(new_h-ch)/2
			cy=max(0,cy-diff); ch=min(fh-cy,new_h)
		else:
			new_w=ch*tgt_as; diff=(new_w-cw)/2
			cx=max(0,cx-diff); cw=min(fw-cx,new_w)
		return (int(cx), int(cy), int(cw), int(ch))

		# fit_scale unused, using cover_scale only
	def draw_pose_overlay(self, vx, vy, vw, vh):
		face_pairs=[(0,1),(0,2),(1,3),(2,4)]
		sh_pairs=[(5,6)]
		idx=np.argmax(self.last_scores)
		pts=self.last_keypoints[idx]
		cx,cy,cw,ch=self.crop_region or (0,0,1,1)
		for a,b in face_pairs+sh_pairs:
			pa,pb=pts[a],pts[b]
			if pa[2]<=self.visibility_threshold or pb[2]<=self.visibility_threshold: continue
			nx,ny=(pa[0]-cx)/cw,(pa[1]-cy)/ch
			if self.mirror_horizontally: nx=1-nx
			if self.rotate_90_degrees:
				px,py=ny*vh,(1-nx)*vw
			else:
				px,py=nx*vw,ny*vh
			fx,fy=int(vx+px),int(vy+py)
			nx2,ny2=(pb[0]-cx)/cw,(pb[1]-cy)/ch
			if self.mirror_horizontally: nx2=1-nx2
			if self.rotate_90_degrees:
				px2,py2=ny2*vh,(1-nx2)*vw
			else:
				px2,py2=nx2*vw,ny2*vh
			fx2,fy2=int(vx+px2),int(vy+py2)
			color=self.green if (a,b) in sh_pairs else self.white
			pygame.draw.line(self.screen,color,(fx,fy),(fx2,fy2),2)
		for j in [0,1,2,3,4,5,6]:
			x,y,v=pts[j]
			if v<=self.visibility_threshold: continue
			nx,ny=(x-cx)/cw,(y-cy)/ch
			if self.mirror_horizontally: nx=1-nx
			if self.rotate_90_degrees:
				px,py=ny*vh,(1-nx)*vw
			else:
				px,py=nx*vw,ny*vh
			fx,fy=int(vx+px),int(vy+py)
			col=self.red if j==0 else (self.white if j<5 else self.green)
			pygame.draw.circle(self.screen,col,(fx,fy),8)

	def handle_events(self):
		for ev in pygame.event.get():
			if ev.type==pygame.QUIT: return False
			if ev.type==pygame.KEYDOWN:
				if ev.key==pygame.K_ESCAPE: return False
				if ev.key==pygame.K_r: self.rotate_90_degrees=not self.rotate_90_degrees
				if ev.key==pygame.K_m: self.mirror_horizontally=not self.mirror_horizontally
				if ev.key==pygame.K_d: self.debug_mode=not self.debug_mode
		return True

	def run(self):
		while self.handle_events():
			self.draw_video_display()
			pygame.display.flip()
			self.clock.tick(30)
		try: self.picam2.stop()
		except: pass
		pygame.quit()

if __name__=='__main__':
	try: FaceShoulderApp().run()
	except KeyboardInterrupt: pass
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		print("Ensure PI + IMX500 + model installed")
