import cv2
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# Configuration parameters for the script
INPUT_PATH        = 'images/input2.jpg'       # Path to the input color image
TILE_GLOB         = 'tiles/*.png'             # Glob pattern to load all tile PNG files
SCALES            = [152, 76, 38]             # Tile sizes (px) for large→small
TARGET_SIZE       = (1064, 1824)              # Output mosaic dimensions (width, height)
MAX_ITERS         = 30000                     # Number of genetic iterations
DISPLAY_EVERY     = 500                       # How often (iterations) to update display
POSTERIZE_THRESH  = 128                       # Threshold for converting grayscale→BW
MASK_BLUR_K       = 201                       # Kernel size for Gaussian blur on mask
MASK_BOOST_ALPHA  = 1.5                       # Contrast boost for detail mask whites
SCALE_THRESHOLDS  = [80, 180]                 # Detail mask mean thresholds for medium/large bins
OVERLAY_ALPHA_MOS = 0.70                      # Opacity of mosaic in overlay
OVERLAY_ALPHA_REF = 0.40                      # Opacity of posterised reference in overlay

# 1. Load and crop the source image, preserving aspect ratio
src_color = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
if src_color is None:
	raise FileNotFoundError(f"Cannot load {INPUT_PATH}")
h_src, w_src = src_color.shape[:2]
# Compute scaling to cover the target size
scale_f = max(TARGET_SIZE[0] / w_src, TARGET_SIZE[1] / h_src)
# Resize and center-crop to exactly TARGET_SIZE
resized = cv2.resize(src_color, (int(w_src*scale_f), int(h_src*scale_f)))
x0 = (resized.shape[1] - TARGET_SIZE[0]) // 2
y0 = (resized.shape[0] - TARGET_SIZE[1]) // 2
target_color = resized[y0:y0+TARGET_SIZE[1], x0:x0+TARGET_SIZE[0]]

# Convert to grayscale and then threshold to a black-and-white posterised image
gray_ref = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)
_, posterised = cv2.threshold(gray_ref, POSTERIZE_THRESH, 255, cv2.THRESH_BINARY)

# 2. Generate a detail mask by edge detection + blur + contrast boost
edges = cv2.Canny(posterised, 100, 200)                       # Detect edges in the BW image
blur = cv2.GaussianBlur(edges, (MASK_BLUR_K, MASK_BLUR_K), 0)  # Smooth edges into regions
mask = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
detail_mask = cv2.convertScaleAbs(mask, alpha=MASK_BOOST_ALPHA, beta=0)  # Boost bright areas

# 3. Load all tile PNGs, composite on white, convert to BW, and build rotations
tiles = []
for path in glob.glob(TILE_GLOB):
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		continue
	# Ensure image has RGBA channels
	if img.ndim == 2:
		b = g = r = img
		a = np.full_like(img, 255)
		img = cv2.merge([b, g, r, a])
	elif img.shape[2] == 3:
		b, g, r = cv2.split(img)
		a = np.full_like(b, 255)
		img = cv2.merge([b, g, r, a])
	# Composite tile onto white background according to alpha
	rgb = img[:, :, :3].astype(float)
	alpha = img[:, :, 3:4].astype(float) / 255.0
	comp = (rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
	# Convert to BW tile
	gray_t = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
	_, bw = cv2.threshold(gray_t, POSTERIZE_THRESH, 255, cv2.THRESH_BINARY)
	tiles.append(bw)
# Build all rotation variants for each BW tile
tile_variants = []
for bw in tiles:
	for angle in (0, 90, 180, 270):
		M = cv2.getRotationMatrix2D((bw.shape[1]/2, bw.shape[0]/2), angle, 1)
		rot = cv2.warpAffine(bw, M, (bw.shape[1], bw.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
		tile_variants.append(rot)
num_variants = len(tile_variants)
console_logs = [f"Loaded {num_variants} tile variants."]

# 4. Determine which grid cells use which tile scale based on mask brightness
regions = []  # list of (row, col, scale)
for scale in SCALES:
	gh = math.ceil(TARGET_SIZE[1] / scale)
	gw = math.ceil(TARGET_SIZE[0] / scale)
	for i in range(gh):
		for j in range(gw):
			y1, x1 = i*scale, j*scale
			y2, x2 = min(y1+scale, TARGET_SIZE[1]), min(x1+scale, TARGET_SIZE[0])
			mval = detail_mask[y1:y2, x1:x2].mean()             # average mask brightness
			# choose scale bin based on brightness thresholds
			if mval >= SCALE_THRESHOLDS[1]: chosen = SCALES[-1]
			elif mval >= SCALE_THRESHOLDS[0]: chosen = SCALES[1]
			else: chosen = SCALES[0]
			# if this cell's scale matches the chosen scale, include region
			if chosen == scale:
				regions.append((i, j, scale))
# 5. Initialize genome: random variant index for each region
genome = [random.randrange(num_variants) for _ in regions]

# Helper to render a variant at given scale
def render_variant(idx, scale):
	var = tile_variants[idx]
	return cv2.resize(var, (scale, scale), interpolation=cv2.INTER_AREA)

# 6. Render initial mosaic and compute initial error
mosaic = np.ones_like(posterised) * 255
for (i, j, s), vidx in zip(regions, genome):
	mosaic[i*s:(i+1)*s, j*s:(j+1)*s] = render_variant(vidx, s)
best_score = np.mean((mosaic.astype(float) - posterised.astype(float))**2)
console_logs.append(f"Initial MSE: {best_score:.2f}")
console_logs.append(f"Distinct variants used: {len(set(genome))}")

# 7. Interactive display: mosaic+posterised overlay, mask, logs
plt.ion()
fig, (ax_l, ax_mask, ax_log) = plt.subplots(1, 3, figsize=(14, 6), gridspec_kw={'width_ratios':[4,1,1]})
for ax in (ax_l, ax_mask, ax_log): ax.axis('off')
# show overlay with configured opacities
overlay = cv2.addWeighted(mosaic, OVERLAY_ALPHA_MOS, posterised, OVERLAY_ALPHA_REF, 0)
im_l = ax_l.imshow(overlay, cmap='gray', vmin=0, vmax=255)
ax_mask.imshow(detail_mask, cmap='gray'); ax_mask.set_title('Detail Mask')
text = ax_log.text(0,1,'\n'.join(console_logs[-10:]), va='top', family='monospace')
ax_log.set_title('Logs')
plt.show()

# 8. Genetic mutation loop: propose & accept local variant swaps
for it in range(1, MAX_ITERS+1):
	idx = random.randrange(len(regions))
	old_vid = genome[idx]
	new_vid = random.randrange(num_variants)
	if new_vid != old_vid:
		i,j,s = regions[idx]
		y1, x1 = i*s, j*s
		y2, x2 = min(y1+s, TARGET_SIZE[1]), min(x1+s, TARGET_SIZE[0])
		old_patch = mosaic[y1:y2, x1:x2]
		new_patch = render_variant(new_vid, s)
		ref = posterised[y1:y2, x1:x2]
		err_old = np.mean((old_patch.astype(float)-ref.astype(float))**2)
		err_new = np.mean((new_patch.astype(float)-ref.astype(float))**2)
		if err_new < err_old:
			genome[idx] = new_vid
			mosaic[y1:y2, x1:x2] = new_patch
			best_score += (err_new - err_old)
	# Update display periodically
	if it % DISPLAY_EVERY == 0 or it == 1:
		overlay = cv2.addWeighted(mosaic, OVERLAY_ALPHA_MOS, posterised, OVERLAY_ALPHA_REF, 0)
		im_l.set_data(overlay)
		text.set_text('\n'.join(console_logs[-10:]+[f"Iter {it}/{MAX_ITERS}, MSE {best_score:.2f}"]))
		fig.canvas.draw_idle(); plt.pause(0.01)

plt.ioff(); plt.show()