"""
core.py -- Satellite change detection engine.

This holds all the actual math from your original two scripts, but
rearranged so nothing is saved to disk in between. Every function takes
arrays in and returns arrays out, which is what lets the Streamlit app
(app.py) run the whole pipeline on uploaded files.

Nothing here knows about Streamlit. You can import these functions from a
plain script too -- see the __main__ block at the bottom for a CLI version.
"""

import base64
import io
from datetime import datetime

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

COLOR_CONSTRUCTION = (255, 0, 0)       # red
COLOR_VEG_LOSS_ONLY = (255, 200, 0)    # amber


# ----------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------

def load_band(source):
    """
    Loads a single-band Sentinel-2 GeoTIFF as a float32 2D array.

    `source` can be a file path OR a file-like object (which is what
    Streamlit's uploader gives you), so the same function works in both
    the app and the command line.
    """
    arr = tifffile.imread(source).astype(np.float32)
    if arr.ndim == 3:
        # Some exports come out with a trailing or leading band axis.
        arr = arr[..., 0] if arr.shape[-1] < arr.shape[0] else arr[0]
    return arr


def load_true_color(source, size=None):
    """Loads the true-color TIFF as an RGB PIL image."""
    arr = tifffile.imread(source)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        img = Image.fromarray(arr[..., :3], mode="RGB")
    return img.resize(size) if size else img


DEG_TO_M = 111320.0  # metres per degree of latitude


def read_geo(source, default_mpp=10.0):
    """
    Pulls georeferencing out of the GeoTIFF header.

    Returns a dict with:
      meters_per_pixel : real ground size of one pixel, in metres
      origin           : (x, y) world coordinate of the top-left pixel
      scale            : (x, y) pixel scale in the file's own units
      is_degrees       : True if the file is lat/long (EPSG:4326)

    This matters more than it looks. A Copernicus Browser export of a small
    area comes out in EPSG:4326, so the pixel scale in the header is in
    DEGREES, roughly 0.00009. Assuming "Sentinel-2 is 10 m/pixel" is wrong
    for these files -- at 28.8 degrees north a pixel is about 8.8 m, so
    every area you report would be ~30% too large.
    """
    geo = {"meters_per_pixel": default_mpp, "origin": None,
           "scale": None, "is_degrees": False}
    try:
        with tifffile.TiffFile(source) as tif:
            page = tif.pages[0]
            scale_tag = page.tags.get("ModelPixelScaleTag")
            tie_tag = page.tags.get("ModelTiepointTag")
            if scale_tag is None or not scale_tag.value:
                return geo

            sx, sy = float(scale_tag.value[0]), float(scale_tag.value[1])
            geo["scale"] = (sx, sy)
            if tie_tag is not None and len(tie_tag.value) >= 6:
                geo["origin"] = (float(tie_tag.value[3]), float(tie_tag.value[4]))

            # A scale below 0.01 can only be degrees -- no projected CRS
            # measures a Sentinel-2 pixel as a hundredth of a metre.
            if sx < 0.01:
                geo["is_degrees"] = True
                lat = geo["origin"][1] if geo["origin"] else 0.0
                metres_x = sx * DEG_TO_M * np.cos(np.radians(lat))
                metres_y = sy * DEG_TO_M
                # Geometric mean, so that width x height gives the true area.
                geo["meters_per_pixel"] = float(np.sqrt(metres_x * metres_y))
            else:
                geo["meters_per_pixel"] = float(np.sqrt(sx * sy))
    except Exception:
        pass
    return geo


def pixel_to_latlon(geo, x, y):
    """
    Converts a pixel column/row into (latitude, longitude).

    Only works when the file is in EPSG:4326, which Copernicus Browser
    exports of small areas are. Returns None otherwise.
    """
    if not geo.get("is_degrees") or not geo.get("origin") or not geo.get("scale"):
        return None
    lon0, lat0 = geo["origin"]
    sx, sy = geo["scale"]
    return (lat0 - y * sy, lon0 + x * sx)


def resize_float(arr, size):
    """Resizes a float32 array to (width, height) with bilinear interpolation."""
    return np.array(Image.fromarray(arr.astype(np.float32)).resize(size, Image.BILINEAR))


# ----------------------------------------------------------------------
# 2. Indices
# ----------------------------------------------------------------------

def _safe_ratio(a, b):
    denom = a + b
    denom = np.where(denom == 0, 1e-6, denom)
    return (a - b) / denom


def compute_indices(red, nir, swir, target_size=None):
    """
    Computes NDVI and NDBI as true float arrays in the range [-1, 1].

    IMPORTANT DIFFERENCE from your old version: these never get squashed
    down into 8-bit PNGs, so the thresholds you set later are in real
    NDVI/NDBI units. Previously a threshold of 0.05 was secretly acting
    like 0.1 because of the (x+1)/2 remap into 0-255.
    """
    ref_size = (red.shape[1], red.shape[0])  # (w, h)

    if nir.shape != red.shape:
        nir = resize_float(nir, ref_size)
    if swir.shape != red.shape:
        # B11 (SWIR) is natively 20 m, so it arrives half-size. Resample up.
        swir = resize_float(swir, ref_size)

    ndvi = _safe_ratio(nir, red)
    ndbi = _safe_ratio(swir, nir)

    if target_size is not None and target_size != ref_size:
        ndvi = resize_float(ndvi, target_size)
        ndbi = resize_float(ndbi, target_size)

    return ndvi, ndbi


# ----------------------------------------------------------------------
# 3. Change detection
# ----------------------------------------------------------------------

def align_to(reference, moving):
    """
    ECC alignment, kept from your original for the case where the two
    dates aren't perfectly co-registered.

    If both TIFFs came from the same Copernicus AOI they are already
    aligned, so app.py leaves this switched off by default -- running it
    on already-aligned images can actually introduce a small error.
    """
    ref_u8 = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mov_u8 = cv2.normalize(moving, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        _, warp = cv2.findTransformECC(ref_u8, mov_u8, warp, cv2.MOTION_EUCLIDEAN, criteria)
        return cv2.warpAffine(
            moving, warp, (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        )
    except cv2.error:
        return moving


def extract_blobs(binary_mask, min_area_px, meters_per_pixel):
    """Removes speckle, then measures every surviving patch."""
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(cleaned)
    blobs = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px >= min_area_px:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(cnt)
            blobs.append({
                "area_px": float(area_px),
                "area_m2": float(area_px) * (meters_per_pixel ** 2),
                "bbox": (int(x), int(y), int(w), int(h)),
                "centroid": (int(x + w // 2), int(y + h // 2)),
            })
    blobs.sort(key=lambda b: b["area_m2"], reverse=True)
    return final_mask, blobs


def detect_changes(ndvi_before, ndvi_after, ndbi_before, ndbi_after, rgb_after,
                   ndvi_drop_threshold=0.05, ndbi_rise_threshold=0.05,
                   min_blob_area_px=100, meters_per_pixel=10.0, do_align=False,
                   geo=None):
    """
    The main event. Returns (overlay_image, construction_blobs,
    veg_loss_blobs, stats).

    Two classes, same as your v4:
      - NDVI fell AND NDBI rose   -> likely new construction
      - NDVI fell but NDBI didn't -> cleared land, not built on yet
    """
    if do_align:
        ndvi_after = align_to(ndvi_before, ndvi_after)
        ndbi_after = align_to(ndbi_before, ndbi_after)

    veg_dropped = (ndvi_before - ndvi_after) > ndvi_drop_threshold
    built_rose = (ndbi_after - ndbi_before) > ndbi_rise_threshold

    construction_raw = np.where(veg_dropped & built_rose, 255, 0).astype(np.uint8)
    veg_loss_raw = np.where(veg_dropped & ~built_rose, 255, 0).astype(np.uint8)

    construction_mask, construction_blobs = extract_blobs(
        construction_raw, min_blob_area_px, meters_per_pixel)
    veg_loss_mask, veg_loss_blobs = extract_blobs(
        veg_loss_raw, min_blob_area_px, meters_per_pixel)

    if geo:
        for b in construction_blobs + veg_loss_blobs:
            b["latlon"] = pixel_to_latlon(geo, *b["centroid"])

    overlay = np.array(rgb_after).copy()
    overlay[veg_loss_mask == 255] = COLOR_VEG_LOSS_ONLY
    overlay[construction_mask == 255] = COLOR_CONSTRUCTION  # drawn on top

    total_px = construction_mask.size
    con_px = int(np.sum(construction_mask == 255))
    veg_px = int(np.sum(veg_loss_mask == 255))
    px_area = meters_per_pixel ** 2

    stats = {
        "num_construction": len(construction_blobs),
        "num_veg_loss": len(veg_loss_blobs),
        "total_construction_area_ha": con_px * px_area / 10000.0,
        "total_veg_loss_area_ha": veg_px * px_area / 10000.0,
        "pct_construction": con_px / total_px * 100,
        "pct_veg_loss": veg_px / total_px * 100,
        "largest_area_m2": construction_blobs[0]["area_m2"] if construction_blobs else 0.0,
    }

    return Image.fromarray(overlay), construction_blobs, veg_loss_blobs, stats


# ----------------------------------------------------------------------
# 4. Presentation
# ----------------------------------------------------------------------

def draw_legend(pil_img):
    """Burns a legend box into the bottom-left of the change map."""
    layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = small = ImageFont.load_default()

    box_w, box_h, margin, sw = 300, 90, 15, 16
    x0, y0 = margin, pil_img.size[1] - box_h - margin
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0, 165))
    draw.text((x0 + 12, y0 + 8), "Legend", font=font, fill=(255, 255, 255, 255))

    draw.rectangle([x0 + 12, y0 + 34, x0 + 28, y0 + 50], fill=COLOR_CONSTRUCTION + (255,))
    draw.text((x0 + 36, y0 + 34), "Likely new construction", font=small, fill=(255, 255, 255, 255))
    draw.rectangle([x0 + 12, y0 + 60, x0 + 28, y0 + 76], fill=COLOR_VEG_LOSS_ONLY + (255,))
    draw.text((x0 + 36, y0 + 60), "Vegetation loss (no building yet)", font=small,
              fill=(255, 255, 255, 255))

    return Image.alpha_composite(pil_img.convert("RGBA"), layer).convert("RGB")


def img_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_html_report(before_img, after_img, change_img, stats, blobs,
                      before_date, after_date, params):
    """Returns the report as an HTML string (app.py offers it as a download)."""
    rows = ""
    for i, b in enumerate(blobs[:15], start=1):
        ll = b.get("latlon")
        coords = f"{ll[0]:.5f}, {ll[1]:.5f}" if ll else "&mdash;"
        rows += (f"<tr><td>{i}</td><td>{b['area_m2']:,.0f}</td>"
                 f"<td>{b['area_m2']/10000:.3f}</td><td>{coords}</td></tr>")

    min_area_m2 = params["min_blob_area_px"] * (params["meters_per_pixel"] ** 2)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Satellite Change Detection Report</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background:#f4f5f7; color:#1a1a1a; margin:0; }}
 .wrap {{ max-width:1000px; margin:0 auto; padding:32px 24px 64px; }}
 h1 {{ font-size:26px; margin-bottom:4px; }}
 .subtitle {{ color:#666; margin:0 0 28px; }}
 .card {{ background:#fff; border-radius:10px; padding:20px 24px; margin-bottom:24px;
         box-shadow:0 1px 3px rgba(0,0,0,.08); }}
 .card h2 {{ font-size:17px; margin-top:0; border-bottom:1px solid #eee; padding-bottom:10px; }}
 .imgs {{ display:flex; gap:14px; flex-wrap:wrap; }}
 .imgs figure {{ margin:0; flex:1; min-width:280px; }}
 .imgs img {{ width:100%; border-radius:6px; display:block; }}
 .imgs figcaption {{ text-align:center; font-size:13px; color:#555; margin-top:6px; }}
 table {{ width:100%; border-collapse:collapse; font-size:14px; }}
 th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #eee; }}
 .stat-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; }}
 .stat-box {{ background:#fafafa; border:1px solid #eee; border-radius:8px; padding:14px; }}
 .stat-box .num {{ font-size:24px; font-weight:700; }}
 .stat-box .label {{ font-size:12px; color:#666; margin-top:2px; }}
 .sw {{ display:inline-block; width:14px; height:14px; border-radius:3px;
        margin-right:6px; vertical-align:middle; }}
 footer {{ text-align:center; color:#999; font-size:12px; margin-top:20px; }}
</style></head><body><div class="wrap">
 <h1>Satellite-Based Change Detection Report</h1>
 <p class="subtitle">NDVI + NDBI dual-index analysis &middot; {before_date} vs {after_date}
   &middot; Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

 <div class="card"><h2>Summary</h2><div class="stat-grid">
  <div class="stat-box"><div class="num">{stats['num_construction']}</div>
    <div class="label">Construction sites detected</div></div>
  <div class="stat-box"><div class="num">{stats['num_veg_loss']}</div>
    <div class="label">Vegetation-loss-only patches</div></div>
  <div class="stat-box"><div class="num">{stats['total_construction_area_ha']:.2f} ha</div>
    <div class="label">Total new-construction area</div></div>
  <div class="stat-box"><div class="num">{stats['pct_construction']:.2f}%</div>
    <div class="label">Of scene flagged as construction</div></div>
  <div class="stat-box"><div class="num">{stats['largest_area_m2']:,.0f} m&sup2;</div>
    <div class="label">Largest single site</div></div>
 </div></div>

 <div class="card"><h2>Legend &amp; Methodology</h2>
  <p><span class="sw" style="background:rgb{COLOR_CONSTRUCTION};"></span>
   <b>Likely new construction</b> &mdash; NDVI fell by more than
   {params['ndvi_drop_threshold']} and NDBI rose by more than
   {params['ndbi_rise_threshold']} at the same pixel.</p>
  <p><span class="sw" style="background:rgb{COLOR_VEG_LOSS_ONLY};"></span>
   <b>Vegetation loss (no building yet)</b> &mdash; NDVI fell but NDBI did not rise.
   Cleared or bare land; possibly early-stage construction.</p>
  <p style="color:#666;font-size:13px;">Cleaned with morphological opening/closing and
   filtered to a minimum patch of {params['min_blob_area_px']} px
   (&asymp; {min_area_m2:,.0f} m&sup2;) at {params['meters_per_pixel']} m/pixel.
   Indices computed as float values in [-1, 1] directly from raw bands.</p></div>

 <div class="card"><h2>Before / After / Change Map</h2><div class="imgs">
  <figure><img src="data:image/png;base64,{before_img}">
    <figcaption>Before &mdash; {before_date}</figcaption></figure>
  <figure><img src="data:image/png;base64,{after_img}">
    <figcaption>After &mdash; {after_date}</figcaption></figure>
  <figure><img src="data:image/png;base64,{change_img}">
    <figcaption>Detected change</figcaption></figure>
 </div></div>

 <div class="card"><h2>Top Construction Sites</h2><table>
  <tr><th>#</th><th>Area (m&sup2;)</th><th>Area (ha)</th><th>Centre (lat, long)</th></tr>
  {rows}</table></div>

 <footer>Sentinel-2 L2A imagery &middot; Copernicus / EU</footer>
</div></body></html>"""


# ----------------------------------------------------------------------
# 5. Command-line mode (your old workflow, still works)
# ----------------------------------------------------------------------

def run_from_folders(before_dir="before", after_dir="after", **params):
    """Reads before/ and after/ folders the way your old scripts did."""
    import glob
    import os

    def find(folder, keyword):
        for p in sorted(glob.glob(os.path.join(folder, "*"))):
            if keyword.lower() in os.path.basename(p).lower():
                return p
        raise FileNotFoundError(f"No file containing '{keyword}' in {folder}/")

    out = {}
    for label, folder in (("before", before_dir), ("after", after_dir)):
        out[label] = {k: find(folder, k) for k in ("B04", "B08", "B11", "true_color")}

    after_red = load_band(out["after"]["B04"])
    target_size = (after_red.shape[1], after_red.shape[0])

    geo = read_geo(out["after"]["B04"])
    params.setdefault("meters_per_pixel", geo["meters_per_pixel"])
    params.setdefault("geo", geo)

    ndvi_b, ndbi_b = compute_indices(
        load_band(out["before"]["B04"]), load_band(out["before"]["B08"]),
        load_band(out["before"]["B11"]), target_size)
    ndvi_a, ndbi_a = compute_indices(
        after_red, load_band(out["after"]["B08"]),
        load_band(out["after"]["B11"]), target_size)

    rgb_after = load_true_color(out["after"]["true_color"], target_size)
    return detect_changes(ndvi_b, ndvi_a, ndbi_b, ndbi_a, rgb_after, **params)


if __name__ == "__main__":
    overlay, con, veg, stats = run_from_folders()
    draw_legend(overlay).save("change_map.png")
    print(f"Construction sites: {stats['num_construction']} "
          f"({stats['total_construction_area_ha']:.2f} ha, "
          f"{stats['pct_construction']:.2f}% of scene)")
    print(f"Vegetation-loss-only patches: {stats['num_veg_loss']}")
    for i, b in enumerate(con[:5], start=1):
        ll = b.get("latlon")
        where = f"{ll[0]:.5f}, {ll[1]:.5f}" if ll else f"pixel {b['centroid']}"
        print(f"  Site {i}: {b['area_m2']:,.0f} m2 at {where}")
    print("Saved: change_map.png")
