#!/usr/bin/env python3
"""
Fast IMX500 Face Detection using Pose Estimation
Minimal processing overhead - mirrors official example performance
Includes --fill-face mode to crop and fill screen with largest detected face
"""

import argparse
import sys
import cv2
import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

# Global variables - same pattern as official example
last_boxes = None
last_scores = None
last_keypoints = None
last_face_crop_data = None
WINDOW_SIZE_H_W = (480, 640)

def ai_output_tensor_parse(metadata: dict):
	"""Parse output tensor - minimally modified from official example."""
	global last_boxes, last_scores, last_keypoints, last_face_crop_data

	np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
	if np_outputs is not None:
		keypoints, scores, boxes = postprocess_higherhrnet(
			outputs=np_outputs,
			img_size=WINDOW_SIZE_H_W,
			img_w_pad=(0, 0),
			img_h_pad=(0, 0),
			detection_threshold=args.detection_threshold,
			network_postprocess=True
		)

		if scores is not None and len(scores) > 0:
			last_keypoints = np.reshape(np.stack(keypoints, axis=0), (len(scores), 17, 3))
			last_boxes = [np.array(b) for b in boxes]
			last_scores = scores

			# Only calculate face crop data if fill-face mode is enabled
			if args.fill_face:
				last_face_crop_data = get_best_face_crop(last_keypoints)
		else:
			last_face_crop_data = None
	else:
		last_face_crop_data = None

	return last_boxes, last_scores, last_keypoints

def get_best_face_crop(keypoints_batch):
	"""Get crop data for the best face (largest/most confident)."""
	if keypoints_batch is None or len(keypoints_batch) == 0:
		return None

	best_face = None
	best_score = 0

	for person_keypoints in keypoints_batch:
		# Check face keypoints (nose, eyes, ears)
		face_indices = [0, 1, 2, 3, 4]
		face_conf_sum = sum(person_keypoints[i][2] for i in face_indices
						  if i < len(person_keypoints) and person_keypoints[i][2] > args.detection_threshold)

		if face_conf_sum > best_score:
			best_score = face_conf_sum
			best_face = person_keypoints

	if best_face is not None:
		return calculate_face_crop_region(best_face)
	return None

def calculate_face_crop_region(keypoints):
	"""Calculate crop region for face fill mode."""
	# Face + shoulder keypoints for better framing
	key_indices = [0, 1, 2, 3, 4, 5, 6]  # nose, eyes, ears, shoulders

	valid_points = []
	for idx in key_indices:
		if idx < len(keypoints):
			x, y, conf = keypoints[idx]
			if conf > args.detection_threshold:
				valid_points.append((int(x), int(y)))

	if len(valid_points) >= 2:
		x_coords = [p[0] for p in valid_points]
		y_coords = [p[1] for p in valid_points]

		min_x, max_x = min(x_coords), max(x_coords)
		min_y, max_y = min(y_coords), max(y_coords)

		# Calculate crop region to fill screen while maintaining aspect ratio
		center_x = (min_x + max_x) // 2
		center_y = (min_y + max_y) // 2

		# Determine crop size based on face size and screen aspect ratio
		face_width = max_x - min_x
		face_height = max_y - min_y

		# Make crop region larger for better framing
		crop_width = max(200, int(face_width * 2.5))
		crop_height = max(150, int(face_height * 2.5))

		# Maintain screen aspect ratio
		screen_ratio = WINDOW_SIZE_H_W[1] / WINDOW_SIZE_H_W[0]  # width/height
		crop_ratio = crop_width / crop_height

		if crop_ratio > screen_ratio:
			# Crop is wider, adjust height
			crop_height = int(crop_width / screen_ratio)
		else:
			# Crop is taller, adjust width
			crop_width = int(crop_height * screen_ratio)

		# Calculate crop bounds
		x1 = max(0, center_x - crop_width // 2)
		y1 = max(0, center_y - crop_height // 2)
		x2 = min(WINDOW_SIZE_H_W[1], x1 + crop_width)
		y2 = min(WINDOW_SIZE_H_W[0], y1 + crop_height)

		# Adjust if crop goes out of bounds
		if x2 == WINDOW_SIZE_H_W[1]:
			x1 = x2 - crop_width
		if y2 == WINDOW_SIZE_H_W[0]:
			y1 = y2 - crop_height

		return (max(0, x1), max(0, y1), min(crop_width, x2-x1), min(crop_height, y2-y1))

	return None

def ai_output_tensor_draw(request: CompletedRequest, boxes, scores, keypoints, stream='main'):
	"""Draw detections - optimized for minimal processing."""
	with MappedArray(request, stream) as m:
		if args.fill_face and last_face_crop_data:
			# Fill face mode - crop and resize
			x, y, w, h = last_face_crop_data

			if w > 0 and h > 0:
				# Extract crop region
				crop = m.array[y:y+h, x:x+w]

				if crop.size > 0:
					# Resize crop to fill entire screen
					resized_crop = cv2.resize(crop, (WINDOW_SIZE_H_W[1], WINDOW_SIZE_H_W[0]))
					m.array[:] = resized_crop

					# Add simple overlay info
					cv2.putText(m.array, 'FACE FILL MODE', (10, 30),
							   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
					cv2.putText(m.array, 'Press ESC to exit', (10, 60),
							   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
					return

		# Normal mode - draw face boxes
		face_count = 0
		if keypoints is not None and len(keypoints) > 0:
			for person_keypoints in keypoints:
				# Quick face detection using keypoints
				face_indices = [0, 1, 2, 3, 4, 5, 6]  # nose, eyes, ears, shoulders
				valid_points = []

				for idx in face_indices:
					if idx < len(person_keypoints):
						x, y, conf = person_keypoints[idx]
						if conf > args.detection_threshold:
							valid_points.append((int(x), int(y)))

				if len(valid_points) >= 2:
					face_count += 1

					# Simple bounding box
					x_coords = [p[0] for p in valid_points]
					y_coords = [p[1] for p in valid_points]

					min_x, max_x = min(x_coords), max(x_coords)
					min_y, max_y = min(y_coords), max(y_coords)

					# Add padding
					pad = 30
					x1 = max(0, min_x - pad)
					y1 = max(0, min_y - pad)
					x2 = min(WINDOW_SIZE_H_W[1], max_x + pad)
					y2 = min(WINDOW_SIZE_H_W[0], max_y + pad)

					# Draw green rectangle
					cv2.rectangle(m.array, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cv2.putText(m.array, f'Person {face_count}', (x1, y1 - 10),
							   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

		# Info display
		info_text = f'People: {face_count}'
		if args.fill_face:
			info_text += ' | Fill mode: ON'

		cv2.putText(m.array, info_text, (10, 30),
				   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def picamera2_pre_callback(request: CompletedRequest):
	"""Main callback - identical to official example structure."""
	boxes, scores, keypoints = ai_output_tensor_parse(request.get_metadata())
	ai_output_tensor_draw(request, boxes, scores, keypoints)

def get_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description="IMX500 Fast Face Detection")
	parser.add_argument("--model", type=str,
					   default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk",
					   help="Path to the pose estimation model")
	parser.add_argument("--detection-threshold", type=float, default=0.3,
					   help="Minimum confidence threshold for detection")
	parser.add_argument("--fill-face", action="store_true",
					   help="Crop and fill screen with largest detected face")
	parser.add_argument("--fps", type=int,
					   help="Override inference rate (default from model)")
	parser.add_argument("--print-intrinsics", action="store_true",
					   help="Print network intrinsics and exit")
	return parser.parse_args()

def get_drawer():
	"""Get drawer for compatibility."""
	categories = intrinsics.labels
	categories = [c for c in categories if c and c != "-"]
	return COCODrawer(categories, imx500, needs_rescale_coords=False)

def main():
	global args, imx500, intrinsics, drawer

	args = get_args()

	# This must be called before instantiation of Picamera2
	imx500 = IMX500(args.model)
	intrinsics = imx500.network_intrinsics
	if not intrinsics:
		intrinsics = NetworkIntrinsics()
		intrinsics.task = "pose estimation"
	elif intrinsics.task != "pose estimation":
		print("Network is not a pose estimation task", file=sys.stderr)
		sys.exit(1)

	# Override intrinsics from args - same as official example
	for key, value in vars(args).items():
		if key == 'labels' and value is not None:
			with open(value, 'r') as f:
				intrinsics.labels = f.read().splitlines()
		elif hasattr(intrinsics, key) and value is not None:
			setattr(intrinsics, key, value)

	# Defaults - exactly like official example
	if intrinsics.inference_rate is None:
		intrinsics.inference_rate = 10
	if intrinsics.labels is None:
		try:
			with open("assets/coco_labels.txt", "r") as f:
				intrinsics.labels = f.read().splitlines()
		except:
			intrinsics.labels = ["person"] * 80
	intrinsics.update_with_defaults()

	if args.print_intrinsics:
		print(intrinsics)
		return

	drawer = get_drawer()

	# Camera setup - identical to official example
	picam2 = Picamera2(imx500.camera_num)
	config = picam2.create_preview_configuration(
		controls={'FrameRate': intrinsics.inference_rate},
		buffer_count=12
	)

	print("Loading pose estimation model on IMX500...")
	if args.fill_face:
		print("Face fill mode enabled - will crop to largest face")

	imx500.show_network_fw_progress_bar()
	picam2.start(config, show_preview=True)
	imx500.set_auto_aspect_ratio()
	picam2.pre_callback = picamera2_pre_callback

	print(f"Face detection started at {intrinsics.inference_rate} fps")
	print("Press Ctrl+C to stop")

	try:
		while True:
			pass
	except KeyboardInterrupt:
		print("\nStopping...")
	finally:
		picam2.stop()
		print("Camera stopped.")

if __name__ == "__main__":
	main()