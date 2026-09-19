"""
app.py -- the web app.

Run it with:    streamlit run app.py

A browser tab opens by itself. Upload the 4 "before" TIFFs and the 4
"after" TIFFs, press Run, and the results appear on the page.

All the actual detection logic lives in core.py -- this file only builds
the interface.
"""

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Satellite Change Detector", layout="wide")


def match_uploads(files, keyword):
    """
    Finds the file, among a batch dropped in together, whose name contains
    `keyword` (case-insensitive) -- same matching rule as the original
    generate_indices.py, so Copernicus's default filenames just work.
    """
    for f in files:
        if keyword.lower() in f.name.lower():
            return f
    return None


# ----------------------------------------------------------------------
# Sidebar: the knobs
# ----------------------------------------------------------------------

st.sidebar.header("Detection settings")

ndvi_drop = st.sidebar.slider(
    "NDVI drop threshold", 0.0, 0.5, 0.10, 0.01,
    help="How much vegetation must disappear before a pixel counts. "
         "Lower = more sensitive, more false positives.")

ndbi_rise = st.sidebar.slider(
    "NDBI rise threshold", 0.0, 0.5, 0.05, 0.01,
    help="How much the built-up signal must increase for a pixel to be "
         "called construction rather than plain clearing.")

min_blob = st.sidebar.slider(
    "Minimum patch size (pixels)", 10, 1000, 100, 10,
    help="Patches smaller than this are treated as noise and discarded.")

auto_scale = st.sidebar.checkbox(
    "Read pixel size from the GeoTIFF", value=True,
    help="Uncheck to set it manually below.")

manual_mpp = st.sidebar.number_input(
    "Metres per pixel", 1.0, 100.0, 10.0, 1.0, disabled=auto_scale)

do_align = st.sidebar.checkbox(
    "Run ECC alignment", value=False,
    help="Only needed if the two dates aren't lined up. Copernicus exports "
         "of the same area already are, and forcing alignment can shift them.")

st.sidebar.divider()
before_date = st.sidebar.text_input("Before date (label only)", "2023-08-17")
after_date = st.sidebar.text_input("After date (label only)", "2026-08-21")


# ----------------------------------------------------------------------
# Main page: uploads
# ----------------------------------------------------------------------

st.title("Satellite-Based Change Detection")
st.caption("Sentinel-2 L2A · NDVI + NDBI dual-index analysis for detecting "
           "new construction and land clearing")

BANDS = ["B04", "B08", "B11", "true_color"]

left, right = st.columns(2)
raw = {}

for column, period, label in ((left, "before", "Before"), (right, "after", "After")):
    with column:
        st.subheader(f"{label} — {before_date if period == 'before' else after_date}")
        raw[period] = st.file_uploader(
            "Drop all 4 files for this date (B04, B08, B11, True Color)",
            type=["tif", "tiff"], accept_multiple_files=True, key=f"{period}_group")

uploads = {"before": {}, "after": {}}
missing = []
for period in ("before", "after"):
    files = raw[period] or []
    for key in BANDS:
        uploads[period][key] = match_uploads(files, key)
        if uploads[period][key] is None:
            missing.append(f"{period}/{key}")

if missing:
    st.info("Select all 4 files per date in one go (⌘/Ctrl-click to multi-select, or "
            "drag them in together). Still missing: " + ", ".join(missing) + ". "
            "From Copernicus Browser's Analytical download panel, export B04, B08, "
            "B11 and True Color as TIFF — filenames don't need renaming, this just "
            "looks for those labels inside each name.")
    st.stop()

if not st.button("Run detection", type="primary", use_container_width=True):
    st.stop()


# ----------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------

with st.spinner("Computing indices and comparing dates..."):
    after_red = core.load_band(uploads["after"]["B04"])
    target_size = (after_red.shape[1], after_red.shape[0])

    uploads["after"]["B04"].seek(0)
    geo = core.read_geo(uploads["after"]["B04"])
    mpp = geo["meters_per_pixel"] if auto_scale else manual_mpp
    uploads["after"]["B04"].seek(0)

    ndvi_before, ndbi_before = core.compute_indices(
        core.load_band(uploads["before"]["B04"]),
        core.load_band(uploads["before"]["B08"]),
        core.load_band(uploads["before"]["B11"]),
        target_size)

    ndvi_after, ndbi_after = core.compute_indices(
        after_red,
        core.load_band(uploads["after"]["B08"]),
        core.load_band(uploads["after"]["B11"]),
        target_size)

    rgb_before = core.load_true_color(uploads["before"]["true_color"], target_size)
    rgb_after = core.load_true_color(uploads["after"]["true_color"], target_size)

    params = {
        "ndvi_drop_threshold": ndvi_drop,
        "ndbi_rise_threshold": ndbi_rise,
        "min_blob_area_px": min_blob,
        "meters_per_pixel": mpp,
    }

    overlay, construction, veg_loss, stats = core.detect_changes(
        ndvi_before, ndvi_after, ndbi_before, ndbi_after, rgb_after,
        do_align=do_align, geo=geo, **params)

    overlay_legend = core.draw_legend(overlay)


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

if auto_scale and geo["is_degrees"]:
    st.success(f"Analysis complete. Pixel size read from the GeoTIFF as "
               f"{mpp:.2f} m (the file is in lat/long, so this was converted "
               f"from degrees — it is not the usual 10 m).")
else:
    st.success(f"Analysis complete at {mpp:g} m/pixel.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Construction sites", stats["num_construction"])
c2.metric("Cleared-land patches", stats["num_veg_loss"])
c3.metric("Construction area", f"{stats['total_construction_area_ha']:.2f} ha")
c4.metric("Largest site", f"{stats['largest_area_m2']:,.0f} m²")

st.divider()

i1, i2, i3 = st.columns(3)
i1.image(rgb_before, caption=f"Before — {before_date}", use_container_width=True)
i2.image(rgb_after, caption=f"After — {after_date}", use_container_width=True)
i3.image(overlay_legend, caption="Detected change", use_container_width=True)

st.caption("Red = vegetation fell and built-up signal rose (likely new construction). "
           "Amber = vegetation fell but no built-up signal (cleared land).")

st.divider()

if construction:
    st.subheader("Detected construction sites")
    rows = []
    for i, b in enumerate(construction, start=1):
        ll = b.get("latlon")
        rows.append({
            "Site": i,
            "Area (m²)": round(b["area_m2"]),
            "Area (ha)": round(b["area_m2"] / 10000, 3),
            "Latitude": round(ll[0], 5) if ll else None,
            "Longitude": round(ll[1], 5) if ll else None,
            "Box (x, y, w, h)": str(b["bbox"]),
        })
    table = pd.DataFrame(rows)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.bar_chart(table.head(15).set_index("Site")["Area (m²)"])

    located = table.dropna(subset=["Latitude", "Longitude"])
    if not located.empty:
        st.subheader("Site locations")
        st.map(located.rename(columns={"Latitude": "lat", "Longitude": "lon"})[["lat", "lon"]])

    st.download_button("Download sites as CSV", table.to_csv(index=False),
                       file_name="detected_sites.csv", mime="text/csv")
else:
    st.warning("No construction sites found. Try lowering the thresholds in the sidebar.")

report = core.build_html_report(
    before_img=core.img_to_base64(rgb_before),
    after_img=core.img_to_base64(rgb_after),
    change_img=core.img_to_base64(overlay_legend),
    stats=stats, blobs=construction,
    before_date=before_date, after_date=after_date, params=params)

st.download_button("Download full HTML report", report,
                   file_name="change_report.html", mime="text/html")
