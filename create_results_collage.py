"""
create_results_collage.py

Builds a single grid image (Degraded Input | Restored (Ours) | Ground Truth)
from the validation samples in frame/{input,GT,results}, for use in the
repository README.

Expected input layout (relative to this script, override with --data_dir):
    frame/input/<id>.npy    degraded LR input, float32, may fall outside [0,1]
    frame/GT/<id>.npy       ground truth, float32, range [0,1]
    frame/results/<id>.png  model output, 16-bit grayscale preview

Only ids present in all three folders are used. Any id missing a GT/result
pair (e.g. a stray/mismatched filename) is skipped and reported.

Usage:
    python create_results_collage.py \
        --data_dir frame \
        --out results_collage.png \
        --max_samples 11
"""

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def find_matched_ids(data_dir):
    inp_dir = os.path.join(data_dir, "input")
    gt_dir = os.path.join(data_dir, "GT")
    res_dir = os.path.join(data_dir, "results")

    inp_ids = {os.path.splitext(f)[0] for f in os.listdir(inp_dir) if f.endswith(".npy")}
    gt_ids = {os.path.splitext(f)[0] for f in os.listdir(gt_dir) if f.endswith(".npy")}
    res_ids = {os.path.splitext(f)[0] for f in os.listdir(res_dir) if f.endswith(".png")}

    matched = sorted(inp_ids & gt_ids & res_ids)
    skipped = sorted((inp_ids | gt_ids | res_ids) - (inp_ids & gt_ids & res_ids))
    if skipped:
        print(f"Skipping {len(skipped)} id(s) without a full input/GT/result triplet: {skipped}")
    return matched


def load_panel(path_npy_or_png, kind, size):
    """Return an 8-bit RGB PIL.Image of shape (size, size) for one panel."""
    if kind == "input":
        arr = np.load(path_npy_or_png).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)  # values may fall outside [0,1] by design; clip for display only
    elif kind == "gt":
        arr = np.load(path_npy_or_png).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
    elif kind == "result":
        im16 = Image.open(path_npy_or_png)
        arr = np.asarray(im16).astype(np.float32) / 65535.0
    else:
        raise ValueError(kind)

    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")
    if img.size != (size, size):
        resample = Image.NEAREST if kind == "input" else Image.BICUBIC
        img = img.resize((size, size), resample=resample)
    return img


def get_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_collage(data_dir, out_path, panel_size=256, max_samples=None, cols=("input", "result", "gt")):
    col_titles = {
        "input": "Degraded Input (128x128, upsampled)",
        "result": "Restored (Ours)",
        "gt": "Ground Truth",
    }

    ids = find_matched_ids(data_dir)
    if max_samples is not None:
        ids = ids[:max_samples]
    if not ids:
        raise RuntimeError("No matched input/GT/result triplets found.")

    n_rows = len(ids)
    n_cols = len(cols)

    pad = 12
    label_col_w = 90
    header_h = 46
    title_h = 56

    cell_w = panel_size
    cell_h = panel_size

    fig_w = label_col_w + n_cols * cell_w + (n_cols + 1) * pad
    fig_h = title_h + header_h + n_rows * cell_h + (n_rows + 1) * pad

    canvas = Image.new("RGB", (fig_w, fig_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(16, bold=True)
    label_font = get_font(14)

    title_text = "KLA PS-01 - Restoration Results (Held-out Validation Samples)"
    title_size = 26
    title_font = get_font(title_size, bold=True)
    max_title_w = fig_w - 2 * pad
    while draw.textlength(title_text, font=title_font) > max_title_w and title_size > 12:
        title_size -= 1
        title_font = get_font(title_size, bold=True)
    tw = draw.textlength(title_text, font=title_font)
    draw.text(((fig_w - tw) / 2, 14), title_text, fill=(20, 20, 20), font=title_font)

    for c, key in enumerate(cols):
        x0 = label_col_w + pad + c * (cell_w + pad)
        text = col_titles[key]
        tw = draw.textlength(text, font=header_font)
        draw.text((x0 + (cell_w - tw) / 2, title_h + 12), text, fill=(30, 30, 30), font=header_font)

    for r, img_id in enumerate(ids):
        y0 = title_h + header_h + pad + r * (cell_h + pad)

        draw.text(
            (10, y0 + cell_h / 2 - 8),
            img_id,
            fill=(60, 60, 60),
            font=label_font,
        )

        paths = {
            "input": os.path.join(data_dir, "input", f"{img_id}.npy"),
            "gt": os.path.join(data_dir, "GT", f"{img_id}.npy"),
            "result": os.path.join(data_dir, "results", f"{img_id}.png"),
        }

        for c, key in enumerate(cols):
            x0 = label_col_w + pad + c * (cell_w + pad)
            panel = load_panel(paths[key], key, panel_size)
            canvas.paste(panel, (x0, y0))
            draw.rectangle(
                [x0, y0, x0 + panel_size - 1, y0 + panel_size - 1],
                outline=(210, 210, 210),
                width=1,
            )

    canvas.save(out_path)
    print(f"Saved collage with {n_rows} sample(s) x {n_cols} column(s) -> {out_path} ({fig_w}x{fig_h}px)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="frame", help="Folder containing input/, GT/, results/")
    parser.add_argument("--out", default="results_collage.png", help="Output collage path")
    parser.add_argument("--panel_size", type=int, default=256, help="Side length of each image panel in px")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of rows (default: all matched)")
    args = parser.parse_args()

    build_collage(args.data_dir, args.out, panel_size=args.panel_size, max_samples=args.max_samples)
