import cv2
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

# Configuration
INPUT_PATH        = 'images/input2.jpg'       # Source image path
TILE_GLOB         = 'tiles/*.png'            # Glob for tile PNGs
SCALES            = [152, 76, 38]            # Tile sizes, large→small
TARGET_SIZE       = (1064, 1824)             # (width, height)
MAX_ITERS         = 30000                    # Genetic iterations
DISPLAY_EVERY     = 500                      # Display interval
POSTERIZE_THRESH  = 128                      # BW threshold
MASK_BLUR_KERNEL  = (501, 501)               # Blur for detail mask
SCALE_THRESHOLDS  = [50, 200]                # Detail mask thresholds to pick scales

# 1. Load & preprocess source
src_color = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
if src_color is None:
	raise FileNotFoundError(f"Cannot load {INPUT_PATH}")
w_src, h_src = src_color.shape[1], src_color.shape[0]
scale_f = max(TARGET_SIZE[0]/w_src, TARGET_SIZE[1]/h_src)
resized = cv2.resize(src_color, (int(w_src*scale_f), int(h_src*scale_f)))
x0 = (resized.shape[1] - TARGET_SIZE[0]) // 2
y0 = (resized.shape[0] - TARGET_SIZE[1]) // 2
target_color = resized[y0:y0+TARGET_SIZE[1], x0:x0+TARGET_SIZE[0]]

# Posterise BW reference
gray_ref = cv2.cvtColor(target_color, cv2.COLOR_BGR2GRAY)
_, posterised = cv2.threshold(gray_ref, POSTERIZE_THRESH, 255, cv2.THRESH_BINARY)

# Detail mask
edges = cv2.Canny(posterised, 100, 200)
blur = cv2.GaussianBlur(edges, MASK_BLUR_KERNEL, 0)
norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
detail_mask = norm  # 0 (low detail) to 255 (high detail)

# 2. Load & prepare tiles (BW arrays)
tiles = []
for p in glob.glob(TILE_GLOB):
	img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
	if img is None:
		continue
	if img.ndim == 2:
		b = g = r = img
		a = np.full_like(img, 255)
		img = cv2.merge([b, g, r, a])
	elif img.shape[2] == 3:
		b, g, r = cv2.split(img)
		a = np.full_like(b, 255)
		img = cv2.merge([b, g, r, a])
	rgb = img[:, :, :3].astype(float)
	alpha = img[:, :, 3:4].astype(float) / 255.0
	white = np.ones_like(rgb) * 255
	comp = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
	gray_t = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
	_, bw = cv2.threshold(gray_t, POSTERIZE_THRESH, 255, cv2.THRESH_BINARY)
	tiles.append(bw)
num_tiles = len(tiles)
console_logs = []
console_logs.append(f"Loaded {num_tiles} tiles.")

# 3. Build regions list based on mask thresholds
regions = []  # (i, j, scale)
for scale in SCALES:
	gh = math.ceil(TARGET_SIZE[1] / scale)
	gw = math.ceil(TARGET_SIZE[0] / scale)
	for i in range(gh):
		for j in range(gw):
			y1, x1 = i*scale, j*scale
			y2 = min((i+1)*scale, TARGET_SIZE[1])
			x2 = min((j+1)*scale, TARGET_SIZE[0])
			mval = detail_mask[y1:y2, x1:x2].mean()
			if mval >= SCALE_THRESHOLDS[1]: chosen = SCALES[-1]
			elif mval >= SCALE_THRESHOLDS[0]: chosen = SCALES[1]
			else: chosen = SCALES[0]
			if chosen == scale:
				regions.append((i, j, scale))
# initialize genome
genome = [(random.randrange(num_tiles), random.randrange(4)) for _ in regions]

# 4. Render and fitness functions
def render_region(mosaic, region, gene):
	i, j, scale = region
	tid, rot = gene
	tile = tiles[tid]
	rs = cv2.resize(tile, (scale, scale), interpolation=cv2.INTER_AREA)
	M = cv2.getRotationMatrix2D((scale/2, scale/2), 90*rot, 1)
	rt = cv2.warpAffine(rs, M, (scale, scale), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
	y1, x1 = i*scale, j*scale
	y2 = min((i+1)*scale, TARGET_SIZE[1])
	x2 = min((j+1)*scale, TARGET_SIZE[0])
	mosaic[y1:y2, x1:x2] = rt[:y2-y1, :x2-x1]

def compute_mse(mos, ref):
	return np.mean((mos.astype(float) - ref.astype(float))**2)

# initial mosaic and score
mosaic = np.ones_like(posterised) * 255
for r, g in zip(regions, genome):
	render_region(mosaic, r, g)
best_score = compute_mse(mosaic, posterised)
console_logs.append(f"Initial MSE: {best_score:.2f}")

# Calculate number of distinct tiles used
used_tiles = set(gene[0] for gene in genome)
console_logs.append(f"Tiles used: {len(used_tiles)} distinct.")

# 5. Interactive display: overlay & previews
plt.ion()
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[4, 1], wspace=0.1, hspace=0.2)
ax_overlay = fig.add_subplot(gs[0:2, 0])
ax_mask = fig.add_subplot(gs[0, 1])
ax_log = fig.add_subplot(gs[1, 1])

# prepare overlay image with swapped opacities
overlay = cv2.addWeighted(mosaic, 0.6, posterised, 0.4, 0)
# draw blue grid at 10% opacity (grid step = smallest scale)
grid_step = SCALES[-1]
h, w = overlay.shape
# overlay grid on image
for x in range(0, w+1, grid_step):
	cv2.line(overlay, (x,0), (x,h), (0,0,255), 1)
for y in range(0, h+1, grid_step):
	cv2.line(overlay, (0,y), (w,y), (0,0,255), 1)
# display overlay (convert BGR grid to gray colormap)
im_overlay = ax_overlay.imshow(overlay, cmap='gray', vmin=0, vmax=255, alpha=1.0)
ax_overlay.set_title('Mosaic Overlay on Posterised')
ax_overlay.axis('off')

# detail mask preview
ax_mask.imshow(detail_mask, cmap='gray')
ax_mask.set_title('Detail Mask')
ax_mask.axis('off')

# console log text
log_text = ax_log.text(0, 1, '\n'.join(console_logs), va='top', family='monospace')
ax_log.axis('off')

plt.show()

# 6. Genetic mutation loop
for it in range(1, MAX_ITERS + 1):
	idx = random.randrange(len(regions))
	old_gene = genome[idx]
	new_gene = (random.randrange(num_tiles), random.randrange(4))
	if new_gene == old_gene:
		continue
	temp = mosaic.copy()
	render_region(temp, regions[idx], new_gene)
	i, j, scale = regions[idx]
	y1, x1 = i*scale, j*scale
	y2 = min(y1+scale, TARGET_SIZE[1])
	x2 = min(x1+scale, TARGET_SIZE[0])
	old_patch = mosaic[y1:y2, x1:x2]
	new_patch = temp[y1:y2, x1:x2]
	ref_patch = posterised[y1:y2, x1:x2]
	err_old = np.mean((old_patch.astype(float) - ref_patch.astype(float))**2)
	err_new = np.mean((new_patch.astype(float) - ref_patch.astype(float))**2)
	if err_new < err_old:
		genome[idx] = new_gene
		mosaic = temp
		best_score += (err_new - err_old)
		console_logs.append(f"Iter {it}: improved MSE to {best_score:.2f}")
	if it % DISPLAY_EVERY == 0:
		overlay = cv2.addWeighted(mosaic, 0.9, posterised, 0.25, 0)
		# redraw grid
		for x in range(0, w+1, grid_step):
			cv2.line(overlay, (x,0), (x,h), (0,0,255), 1)
		for y in range(0, h+1, grid_step):
			cv2.line(overlay, (0,y), (w,y), (0,0,255), 1)
		im_overlay.set_data(overlay)
		log_text.set_text('\n'.join(console_logs[-20:]))
		fig.canvas.draw_idle()
		plt.pause(0.01)

plt.ioff()
plt.show()
