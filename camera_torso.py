#!/usr/bin/env python3
import os
# 1) Silence libcamera commit messages
os.environ["LIBCAMERA_LOG_LEVEL"] = "ERROR"

import cv2
from picamera2 import Picamera2, MappedArray
from picamera2.previews import DrmPreview
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_nanodet_detection

# ── CONFIGURATION ───────────────────────────────────────────────────
MODEL_PATH     = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
CONF_THRESH    = 0.5   # minimum confidence
IOU_THRESH     = 0.5   # NMS IoU threshold
MAX_RAW_DETS   = 10    # raw outputs before NMS
MAX_DRAW_DETS  = 2     # draw up to two boxes

# ── INITIALIZE IMX500 NPU ────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
	raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
imx500     = IMX500(MODEL_PATH)
intrinsics = imx500.network_intrinsics
print("✅ Loaded model:", MODEL_PATH)
print("ℹ️  Labels:", intrinsics.labels)

# ── SETUP Picamera2 STREAM ───────────────────────────────────────────
picam2 = Picamera2(imx500.camera_num)
cfg = picam2.create_preview_configuration(
	main     ={"format":"RGB888","size":(320,240)},
	controls ={"FrameRate": intrinsics.inference_rate}
)
picam2.configure(cfg)
picam2.start_preview(DrmPreview())  # direct to HDMI
picam2.start()
print("🚀 Camera started; entering loop")

try:
	frame_idx = 0
	while True:
		# 2) Capture aligned frame+metadata
		req  = picam2.capture_request(wait=True)
		meta = req.get_metadata()

		# 3) Copy out frame and release DMA
		with MappedArray(req, "main") as m:
			frame = m.array.copy()
		req.release()

		# 4) Run on-chip inference
		outs = imx500.get_outputs(meta, add_batch=True)
		dets = []
		if outs:
			out0 = outs[0]
			# guard against empty outputs
			if out0 is not None and getattr(out0, "size", 0) > 0:
				try:
					raw = postprocess_nanodet_detection(
						outputs     = out0,
						conf        = CONF_THRESH,
						iou_thres   = IOU_THRESH,
						max_out_dets= MAX_RAW_DETS
					)[0]
				except Exception as e:
					print("⚠️ postprocess error:", e)
					raw = []
			else:
				raw = []
		else:
			raw = []

		# 5) Filter “person” and convert coords
		for box, score, cls in raw:
			if score < CONF_THRESH: continue
			if intrinsics.labels[int(cls)] != "person": continue
			x, y, w, h = imx500.convert_inference_coords(box, meta, picam2)
			dets.append((int(x), int(y), int(w), int(h)))
			if len(dets) >= MAX_DRAW_DETS:
				break

		# 6) Draw and report
		for (x, y, w, h) in dets:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		print(f"Frame {frame_idx}: drew {len(dets)} person(s)")
		frame_idx += 1

		# 7) Display on HDMI
		cv2.imshow("AI Torso Detection", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

except KeyboardInterrupt:
	print("\n🛑 Interrupted by user")

finally:
	cv2.destroyAllWindows()
	picam2.stop_preview()
	picam2.stop()
	print("✅ Clean exit")
