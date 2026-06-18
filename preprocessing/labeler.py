"""
Arthrogryposis ROM — Frame Labeling Tool
=========================================
Standalone OpenCV-based labeling tool.
Saves output in DeepLabCut-compatible CSV + H5 format.

Dependencies (all lightweight, no DLC GUI required):
    pip install opencv-python numpy pandas pyyaml h5py

Usage:
    python labeler.py --frames "C:/path/to/files/name" --scorer "Write your name here" --output "C:/path/to/output"

TRUNK MIDPOINT NOTE:
    This tool does NOT ask you to label trunk_midpoint directly.
    It is computed automatically from your labeled shoulder and hip points:

        trunk_midpoint = shoulder_midpoint shifted 20% toward hip_midpoint

    This gives a point just below the shoulder line at roughly sternum/upper
    chest level — more anatomically accurate than the pure shoulder midpoint
    and fully consistent across all frames and labelers.

    The computed point is saved alongside your manual labels in the output CSV.
"""

import cv2
import numpy as np
import pandas as pd
import yaml
import os
import sys
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# KEYPOINTS
# These are the points your team manually labels.
# trunk_midpoint is NOT in this list — it is computed automatically.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BODYPARTS = [
    "chin",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_thumbmcp",
    "right_thumbtip",
    "right_indexmcp",
    "right_indextip",
    "right_middlemcp",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_thumbmcp",
    "left_thumbtip",
    "left_indexmcp",
    "left_indextip",
    "left_middlemcp"
]

# ─────────────────────────────────────────────────────────────────────────────
# TRUNK MIDPOINT SETTINGS
# trunk_midpoint = shoulder_midpoint + TRUNK_DROP_RATIO * (hip_midpoint - shoulder_midpoint)
#
# TRUNK_DROP_RATIO = 0.0  → exactly between the two shoulders
# TRUNK_DROP_RATIO = 0.2  → 20% of the way from shoulders toward hips
#                           (roughly sternum/upper chest level) ← recommended
# TRUNK_DROP_RATIO = 0.5  → halfway between shoulders and hips
#
# We use 0.2 as the default. Change this value if your clinicians prefer
# a different trunk reference point.
# ─────────────────────────────────────────────────────────────────────────────

TRUNK_DROP_RATIO = 0.2

# Which keypoints are used to compute trunk_midpoint
TRUNK_REQUIRES = {
    "shoulders": ("right_shoulder", "left_shoulder"),
    "hips":      ("right_hip",      "left_hip"),      # optional — see below
}

# NOTE: right_hip and left_hip are NOT in DEFAULT_BODYPARTS above because
# your current keypoint set does not include them. The tool will compute
# trunk_midpoint using shoulders only if hips are missing, using
# TRUNK_DROP_RATIO=0.0 in that fallback case.
# If you add right_hip and left_hip to DEFAULT_BODYPARTS, the full
# shoulder-to-hip drop computation activates automatically.

# ─────────────────────────────────────────────────────────────────────────────
# COLORS — one per keypoint (BGR), must match DEFAULT_BODYPARTS order
# ─────────────────────────────────────────────────────────────────────────────
COLORS = [
    (0,   255, 100),   # chin             — bright green
    (0,   200, 255),   # right_shoulder   — yellow
    (0,   120, 255),   # right_elbow      — orange
    (0,   60,  255),   # right_wrist      — deep orange
    (60,  0,   255),   # right_thumbmcp   — red-purple
    (120, 0,   255),   # right_thumbtip   — purple
    (180, 0,   255),   # right_indexmcp   — violet
    (220, 0,   200),   # right_indextip   — magenta
    (255, 0,   150),   # right_middlemcp  — pink
    (255, 100, 0),     # left_shoulder    — blue
    (255, 180, 0),     # left_elbow       — cyan
    (255, 230, 0),     # left_wrist       — light cyan
    (200, 255, 0),     # left_thumbmcp    — lime
    (100, 255, 0),     # left_thumbtip    — green
    (0,   255, 60),    # left_indexmcp    — teal-green
    (0,   255, 180),   # left_indextip    — teal
    (0,   200, 200)   # left_middlemcp   — dark teal
    
]

# Color for the computed trunk_midpoint overlay (not labeled — shown as reference)
TRUNK_MIDPOINT_COLOR = (255, 255, 0)   # bright yellow

DOT_RADIUS  = 6
FONT        = cv2.FONT_HERSHEY_SIMPLEX
PANEL_WIDTH = 280


# ─────────────────────────────────────────────────────────────────────────────
# TRUNK MIDPOINT COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_trunk_midpoint(labels, drop_ratio=TRUNK_DROP_RATIO):
    """
    Compute the trunk midpoint from labeled shoulder (and optionally hip) points.

    Returns (x, y) tuple or None if shoulders are not both labeled.

    Logic:
        shoulder_mid = midpoint(right_shoulder, left_shoulder)
        if hips available:
            hip_mid = midpoint(right_hip, left_hip)
            trunk_mid = shoulder_mid + drop_ratio * (hip_mid - shoulder_mid)
        else:
            trunk_mid = shoulder_mid   (pure midpoint, no drop)
    """
    rs = labels.get("right_shoulder")
    ls = labels.get("left_shoulder")

    if rs is None or ls is None:
        return None   # cannot compute without both shoulders

    # Shoulder midpoint
    sx = (rs[0] + ls[0]) / 2.0
    sy = (rs[1] + ls[1]) / 2.0

    # Try to get hip midpoint for the drop
    rh = labels.get("right_hip")
    lh = labels.get("left_hip")

    if rh is not None and lh is not None:
        hx = (rh[0] + lh[0]) / 2.0
        hy = (rh[1] + lh[1]) / 2.0
        tx = sx + drop_ratio * (hx - sx)
        ty = sy + drop_ratio * (hy - sy)
    else:
        # Hips not labeled — use pure shoulder midpoint
        # (drop_ratio has no anchor, so we ignore it)
        tx, ty = sx, sy

    return (tx, ty)


def add_computed_points(labels):
    """
    Add automatically computed points to a label dict.
    These are saved to CSV alongside manual labels but never shown
    as points the user needs to click.
    Returns a new dict with computed points added.
    """
    out = dict(labels)
    tp = compute_trunk_midpoint(labels)
    if tp is not None:
        out["trunk_midpoint"] = list(tp)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clamp_color(idx):
    return COLORS[idx % len(COLORS)]


def draw_sidebar(canvas, bodyparts, current_bp_idx, labels, img_h):
    """Draw the keypoint list panel on the right side of the canvas."""
    cv2.rectangle(canvas,
                  (canvas.shape[1] - PANEL_WIDTH, 0),
                  (canvas.shape[1], img_h),
                  (30, 30, 30), -1)

    cv2.putText(canvas, "KEYPOINTS", (canvas.shape[1] - PANEL_WIDTH + 10, 28),
                FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(canvas,
             (canvas.shape[1] - PANEL_WIDTH + 8, 36),
             (canvas.shape[1] - 8, 36),
             (80, 80, 80), 1)

    for i, bp in enumerate(bodyparts):
        y       = 56 + i * 34
        color   = clamp_color(i)
        labeled = labels.get(bp) is not None

        # Highlight current
        if i == current_bp_idx:
            cv2.rectangle(canvas,
                          (canvas.shape[1] - PANEL_WIDTH + 4, y - 13),
                          (canvas.shape[1] - 4, y + 9),
                          (60, 60, 100), -1)

        cv2.circle(canvas,
                   (canvas.shape[1] - PANEL_WIDTH + 18, y - 2),
                   5, color, -1 if labeled else 1)

        status = "v" if labeled else " "
        text   = f"{status} {bp}"
        cv2.putText(canvas, text,
                    (canvas.shape[1] - PANEL_WIDTH + 30, y + 4),
                    FONT, 0.52,
                    (220, 220, 220) if i == current_bp_idx else (160, 160, 160),
                    1, cv2.LINE_AA)

    # Computed trunk midpoint indicator
    tp = compute_trunk_midpoint(labels)
    trunk_y = 56 + len(bodyparts) * 26 + 8
    cv2.line(canvas,
             (canvas.shape[1] - PANEL_WIDTH + 8, trunk_y - 6),
             (canvas.shape[1] - 8, trunk_y - 6),
             (80, 80, 80), 1)
    trunk_status = "AUTO" if tp is not None else "need shoulders"
    cv2.circle(canvas,
               (canvas.shape[1] - PANEL_WIDTH + 18, trunk_y + 6),
               5, TRUNK_MIDPOINT_COLOR if tp is not None else (80, 80, 80), -1)
    cv2.putText(canvas, f"  trunk_midpoint [{trunk_status}]",
                (canvas.shape[1] - PANEL_WIDTH + 30, trunk_y + 10),
                FONT, 0.36,
                TRUNK_MIDPOINT_COLOR if tp is not None else (80, 80, 80),
                1, cv2.LINE_AA)

    # Controls
    help_y = img_h - 195
    cv2.line(canvas,
             (canvas.shape[1] - PANEL_WIDTH + 8, help_y - 8),
             (canvas.shape[1] - 8, help_y - 8),
             (80, 80, 80), 1)
    tips = [
        "CONTROLS",
        "LClick  : place point",
        "Del/r   : remove point",
        "Tab/n   : next keypoint",
        "N       : prev keypoint",
        "d / ->  : next frame",
        "a / <-  : prev frame",
        "s       : save progress",
        "h       : skip frame",
        "q / Esc : quit & save",
    ]
    for j, tip in enumerate(tips):
        cv2.putText(canvas, tip,
                    (canvas.shape[1] - PANEL_WIDTH + 10, help_y + j * 18),
                    FONT, 0.35,
                    (200, 200, 200) if j == 0 else (130, 130, 130),
                    1, cv2.LINE_AA)


def draw_labels_on_image(img, labels, bodyparts, current_bp_idx, zoom_factor=1.0):
    """Draw all placed keypoint dots and the computed trunk midpoint."""
    out = img.copy()

    # Draw manual keypoints
    for i, bp in enumerate(bodyparts):
        pt = labels.get(bp)
        if pt is None:
            continue
        x, y   = int(pt[0] * zoom_factor), int(pt[1] * zoom_factor)
        color  = clamp_color(i)
        radius = max(3, int(DOT_RADIUS * zoom_factor))
        thick  = 2 if i == current_bp_idx else 1

        cv2.circle(out, (x, y), radius, color, thick)
        cv2.circle(out, (x, y), 2, (255, 255, 255), -1)



    # Draw computed trunk midpoint (yellow diamond — visually distinct from manual points)
    tp = compute_trunk_midpoint(labels)
    if tp is not None:
        tx = int(tp[0] * zoom_factor)
        ty = int(tp[1] * zoom_factor)
        size = max(5, int(7 * zoom_factor))

        # Draw diamond shape
        pts = np.array([
            [tx,        ty - size],
            [tx + size, ty       ],
            [tx,        ty + size],
            [tx - size, ty       ],
        ], dtype=np.int32)
        cv2.fillPoly(out, [pts], TRUNK_MIDPOINT_COLOR)
        cv2.polylines(out, [pts], True, (0, 0, 0), 1)

        # Draw lines from each shoulder to trunk midpoint
        rs = labels.get("right_shoulder")
        ls = labels.get("left_shoulder")
        if rs:
            cv2.line(out,
                     (int(rs[0]*zoom_factor), int(rs[1]*zoom_factor)),
                     (tx, ty),
                     (TRUNK_MIDPOINT_COLOR[0]//2, TRUNK_MIDPOINT_COLOR[1]//2, 0),
                     1, cv2.LINE_AA)
        if ls:
            cv2.line(out,
                     (int(ls[0]*zoom_factor), int(ls[1]*zoom_factor)),
                     (tx, ty),
                     (TRUNK_MIDPOINT_COLOR[0]//2, TRUNK_MIDPOINT_COLOR[1]//2, 0),
                     1, cv2.LINE_AA)



    return out


def draw_status_bar(canvas, frame_idx, total_frames, n_labeled, n_total_bp,
                    img_path, current_bp):
    bar_h = 38
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (20, 20, 40), -1)
    fname    = Path(img_path).name
    pct      = int(100 * n_labeled / n_total_bp) if n_total_bp else 0
    bar_text = (f"Frame {frame_idx + 1}/{total_frames}  |  "
                f"{fname}  |  "
                f"Labeled: {n_labeled}/{n_total_bp} ({pct}%)  |  "
                f"Active: {current_bp}")
    cv2.putText(canvas, bar_text, (10, 25),
                FONT, 0.43, (200, 220, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_json(all_labels, save_path):
    with open(save_path, "w") as f:
        json.dump(all_labels, f, indent=2)


def load_json(save_path):
    if os.path.exists(save_path):
        with open(save_path) as f:
            return json.load(f)
    return {}


def export_dlc_csv(all_labels, bodyparts, scorer, output_dir):
    """
    Export labels in DeepLabCut-compatible multi-index CSV format.
    Includes both manually labeled points AND the computed trunk_midpoint.
    """
    # Full list of exported points = manual + computed
    all_bodyparts = list(bodyparts) + ["trunk_midpoint"]

    rows      = []
    img_paths = sorted(all_labels.keys())

    for img_path in img_paths:
        frame_labels = all_labels[img_path]
        # Add computed trunk midpoint to this frame's labels
        enriched = add_computed_points(frame_labels)
        row = {}
        for bp in all_bodyparts:
            pt = enriched.get(bp)
            if pt is not None:
                row[(scorer, bp, "x")] = pt[0]
                row[(scorer, bp, "y")] = pt[1]
            else:
                row[(scorer, bp, "x")] = np.nan
                row[(scorer, bp, "y")] = np.nan
        rows.append((img_path, row))

    index      = pd.Index([r[0] for r in rows], name="image")
    col_tuples = [(scorer, bp, coord)
                  for bp in all_bodyparts for coord in ["x", "y"]]
    columns    = pd.MultiIndex.from_tuples(col_tuples,
                                           names=["scorer", "bodyparts", "coords"])

    data = np.full((len(rows), len(col_tuples)), np.nan)
    for ri, (_, row) in enumerate(rows):
        for ci, col in enumerate(col_tuples):
            if col in row:
                data[ri, ci] = row[col]

    df = pd.DataFrame(data, index=index, columns=columns)

    csv_path = os.path.join(output_dir, f"CollectedData_{scorer}.csv")
    df.to_csv(csv_path)
    print(f"\n[SAVED] CSV  -> {csv_path}")
    print(f"        Columns: {len(all_bodyparts)} keypoints "
          f"({len(bodyparts)} manual + 1 computed trunk_midpoint)")

    try:
        h5_path = os.path.join(output_dir, f"CollectedData_{scorer}.h5")
        df.to_hdf(h5_path, key="df_with_missing", mode="w")
        print(f"[SAVED] H5   -> {h5_path}")
    except Exception as e:
        print(f"[WARN]  H5 save failed: {e}")
        print("        CSV export is sufficient for DLC training.")

    return df


def export_summary(all_labels, bodyparts, output_dir):
    total_frames   = len(all_labels)
    labeled_counts = {bp: 0 for bp in bodyparts}
    trunk_computed = 0

    for frame_labels in all_labels.values():
        for bp in bodyparts:
            if frame_labels.get(bp) is not None:
                labeled_counts[bp] += 1
        if compute_trunk_midpoint(frame_labels) is not None:
            trunk_computed += 1

    print("\n" + "="*58)
    print("  LABELING SUMMARY")
    print("="*58)
    print(f"  Total frames labeled : {total_frames}")
    print(f"  {'Keypoint':<24}  {'Labeled':>7}  {'Coverage':>9}")
    print("  " + "-"*46)
    for bp, count in labeled_counts.items():
        pct = 100 * count / total_frames if total_frames else 0
        print(f"  {bp:<24}  {count:>7}  {pct:>8.1f}%")
    print("  " + "-"*46)
    pct = 100 * trunk_computed / total_frames if total_frames else 0
    print(f"  {'trunk_midpoint [AUTO]':<24}  {trunk_computed:>7}  {pct:>8.1f}%")
    print("="*58)

    summary = {
        "total_frames":       total_frames,
        "keypoint_coverage":  labeled_counts,
        "trunk_midpoint_auto_computed": trunk_computed,
        "trunk_drop_ratio":   TRUNK_DROP_RATIO,
        "exported_at":        datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "labeling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LABELING LOOP
# ─────────────────────────────────────────────────────────────────────────────

class Labeler:
    def __init__(self, image_paths, bodyparts, scorer, output_dir,
                 existing_labels=None):
        self.image_paths = image_paths
        self.bodyparts   = bodyparts
        self.scorer      = scorer
        self.output_dir  = output_dir
        self.json_path   = os.path.join(output_dir, "labels_progress.json")
        self.all_labels  = existing_labels or load_json(self.json_path)
        self.frame_idx   = 0
        self.bp_idx      = 0
        self.zoom        = 1.0

        # Resume from first unlabeled frame
        for i, p in enumerate(image_paths):
            if p not in self.all_labels:
                self.frame_idx = i
                break

    def mouse_callback(self, event, x, y, flags, param):
        img_w         = param["img_w"]

        if x >= int(img_w * self.zoom):   # ignore sidebar clicks
            return
        if y < 38:                         # ignore status bar clicks
            return

        actual_y = y - 38

        if event == cv2.EVENT_LBUTTONDOWN:
            ox = int(x / self.zoom)
            oy = int(actual_y / self.zoom)
            bp         = self.bodyparts[self.bp_idx]
            frame_path = self.image_paths[self.frame_idx]

            if frame_path not in self.all_labels:
                self.all_labels[frame_path] = {}
            self.all_labels[frame_path][bp] = [ox, oy]

            # Auto-advance to next unlabeled bodypart
            labeled = self.all_labels[frame_path]
            for offset in range(1, len(self.bodyparts)):
                next_idx = (self.bp_idx + offset) % len(self.bodyparts)
                if labeled.get(self.bodyparts[next_idx]) is None:
                    self.bp_idx = next_idx
                    break

    def render(self, img_orig):
        frame_path = self.image_paths[self.frame_idx]
        labels     = self.all_labels.get(frame_path, {})

        h, w   = img_orig.shape[:2]
        disp_w = int(w * self.zoom)
        disp_h = int(h * self.zoom)
        img_disp = cv2.resize(img_orig, (disp_w, disp_h))
        img_disp = draw_labels_on_image(
            img_disp, labels, self.bodyparts, self.bp_idx, self.zoom)

        n_labeled = sum(1 for v in labels.values() if v is not None)
        sidebar_h = max(disp_h, len(self.bodyparts) * 26 + 320)
        canvas    = np.zeros((38 + sidebar_h, disp_w + PANEL_WIDTH, 3), dtype=np.uint8)
        canvas[38:38 + disp_h, :disp_w, :] = img_disp

        draw_status_bar(canvas, self.frame_idx, len(self.image_paths),
                        n_labeled, len(self.bodyparts),
                        frame_path, self.bodyparts[self.bp_idx])
        draw_sidebar(canvas, self.bodyparts, self.bp_idx, labels, sidebar_h)
        return canvas, w, disp_h

    def run(self):
        win_name = "Arthrogryposis ROM Labeler  |  q = quit & save"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        print("\n" + "="*58)
        print("  LABELING TOOL STARTED")
        print("="*58)
        print(f"  Frames        : {len(self.image_paths)}")
        print(f"  Manual KPs    : {len(self.bodyparts)}")
        print(f"  Auto computed : trunk_midpoint "
              f"(drop ratio: {TRUNK_DROP_RATIO})")
        print(f"  Scorer        : {self.scorer}")
        print(f"  Output        : {self.output_dir}")
        print("="*58)
        print("  LClick=place  Del/r=remove  Tab=next BP")
        print("  d/right=next frame  a/left=prev frame")
        print("  s=save  h=skip  q=quit & save")
        print("="*58 + "\n")

        while True:
            img_orig = cv2.imread(self.image_paths[self.frame_idx])
            if img_orig is None:
                print(f"[WARN] Could not read: {self.image_paths[self.frame_idx]}")
                self.frame_idx = min(self.frame_idx + 1, len(self.image_paths) - 1)
                continue

            canvas, img_w, img_h_display = self.render(img_orig)
            cv2.setMouseCallback(win_name, self.mouse_callback,
                                 {"img_w": img_w, "img_h_display": img_h_display})
            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key in (ord('q'), 27):
                print("\n[INFO] Saving and quitting...")
                break
            elif key in (ord('d'), 83):
                self.frame_idx = min(self.frame_idx + 1, len(self.image_paths) - 1)
                self.bp_idx    = 0
            elif key in (ord('a'), 81):
                self.frame_idx = max(self.frame_idx - 1, 0)
                self.bp_idx    = 0
            elif key in (9, ord('n')):           # Tab or n
                self.bp_idx = (self.bp_idx + 1) % len(self.bodyparts)
            elif key == ord('N'):                # Shift+N
                self.bp_idx = (self.bp_idx - 1) % len(self.bodyparts)
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(self.bodyparts):
                    self.bp_idx = idx
            elif key in (127, 8, ord('r')):      # Del, Backspace, r
                bp         = self.bodyparts[self.bp_idx]
                frame_path = self.image_paths[self.frame_idx]
                if frame_path in self.all_labels:
                    self.all_labels[frame_path].pop(bp, None)
            elif key == ord('h'):
                fp = self.image_paths[self.frame_idx]
                if fp not in self.all_labels:
                    self.all_labels[fp] = {}
                self.all_labels[fp]["__skipped__"] = True
                self.frame_idx = min(self.frame_idx + 1, len(self.image_paths) - 1)
            elif key == ord('s'):
                save_json(self.all_labels, self.json_path)
                print(f"[SAVED] Progress -> {self.json_path}")
            elif key in (ord('+'), ord('=')):
                self.zoom = min(self.zoom + 0.1, 3.0)
            elif key == ord('-'):
                self.zoom = max(self.zoom - 0.1, 0.3)
            elif key == ord('g'):
                cv2.destroyAllWindows()
                try:
                    target = int(input("  Go to frame number: ")) - 1
                    self.frame_idx = max(0, min(target, len(self.image_paths) - 1))
                except ValueError:
                    pass
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        cv2.destroyAllWindows()
        self._finish()

    def _finish(self):
        os.makedirs(self.output_dir, exist_ok=True)
        save_json(self.all_labels, self.json_path)

        clean_labels = {
            k: {bp: v for bp, v in labels.items() if bp != "__skipped__"}
            for k, labels in self.all_labels.items()
            if labels and not labels.get("__skipped__")
        }

        if clean_labels:
            export_dlc_csv(clean_labels, self.bodyparts, self.scorer, self.output_dir)
            export_summary(clean_labels, self.bodyparts, self.output_dir)
        else:
            print("[WARN] No labels to export yet.")

        print(f"\n[DONE] All files saved to: {self.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# SETUP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames_from_videos(video_paths, output_dir, n_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for vp in video_paths:
        cap   = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            print(f"[WARN] Could not read: {vp}")
            cap.release()
            continue
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        vname   = Path(vp).stem
        print(f"  Extracting {len(indices)} frames from {Path(vp).name} ...")
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            fpath = os.path.join(output_dir, f"{vname}_frame{idx:05d}.png")
            cv2.imwrite(fpath, frame)
            saved_paths.append(fpath)
        cap.release()
    print(f"[INFO] {len(saved_paths)} frames extracted to {output_dir}")
    return saved_paths


def load_bodyparts_from_dlc_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("bodyparts", DEFAULT_BODYPARTS)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Arthrogryposis ROM Frame Labeling Tool"
    )
    parser.add_argument("--frames",    type=str, default=None)
    parser.add_argument("--videos",    type=str, default=None)
    parser.add_argument("--project",   type=str, default=None)
    parser.add_argument("--scorer",    type=str, default="Researcher")
    parser.add_argument("--output",    type=str, default="./labeled_data")
    parser.add_argument("--nframes",   type=int, default=30)
    parser.add_argument("--bodyparts", type=str, default=None)
    global TRUNK_DROP_RATIO  # ← moved here, before any reference to the variable

    parser.add_argument("--trunk_drop", type=float, default=TRUNK_DROP_RATIO,
                        help=f"Trunk drop ratio 0.0-1.0 (default {TRUNK_DROP_RATIO})")
    args = parser.parse_args()

    TRUNK_DROP_RATIO = args.trunk_drop

    print("\n" + "="*58)
    print("  ARTHROGRYPOSIS ROM — LABELING TOOL")
    print("="*58)
    print(f"  Trunk midpoint drop ratio: {TRUNK_DROP_RATIO}")
    print(f"  (0.0 = pure shoulder midpoint, 0.2 = 20% toward hips)")

    # Determine bodyparts
    if args.bodyparts:
        bodyparts = [b.strip() for b in args.bodyparts.split(",")]
    elif args.project:
        config_path = os.path.join(args.project, "config.yaml")
        if os.path.exists(config_path):
            bodyparts = load_bodyparts_from_dlc_config(config_path)
            print(f"[INFO] Loaded {len(bodyparts)} keypoints from config.yaml")
        else:
            bodyparts = DEFAULT_BODYPARTS
    else:
        bodyparts = DEFAULT_BODYPARTS

    print(f"[INFO] Keypoints ({len(bodyparts)}): {', '.join(bodyparts)}")
    print(f"[INFO] + trunk_midpoint (auto-computed, not manually labeled)")

    output_dir = args.output
    if args.project:
        output_dir = os.path.join(args.project, "labeled_data", "labeling_session")
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []

    if args.frames:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            image_paths += glob.glob(os.path.join(args.frames, ext))
        image_paths = sorted(image_paths)
        print(f"[INFO] Found {len(image_paths)} frames in {args.frames}")

    elif args.videos:
        video_list = [v.strip() for v in args.videos.split(",")]
        frames_dir = os.path.join(output_dir, "extracted_frames")
        image_paths = extract_frames_from_videos(video_list, frames_dir, args.nframes)

    elif args.project:
        frames_dir = os.path.join(args.project, "labeled-data")
        for ext in ("*.png", "*.jpg"):
            image_paths += glob.glob(os.path.join(frames_dir, "**", ext), recursive=True)
        image_paths = sorted(image_paths)
        if not image_paths:
            videos_dir = os.path.join(args.project, "videos")
            if os.path.exists(videos_dir):
                vids = (glob.glob(os.path.join(videos_dir, "*.mp4")) +
                        glob.glob(os.path.join(videos_dir, "*.avi")) +
                        glob.glob(os.path.join(videos_dir, "*.mov")))
                frames_dir = os.path.join(output_dir, "extracted_frames")
                image_paths = extract_frames_from_videos(vids, frames_dir, args.nframes)

    else:
        print("\n  No input specified. Running in interactive mode.")
        print("  1) Label frames from a folder of images")
        print("  2) Extract frames from video files, then label")
        choice = input("\n  Enter 1 or 2: ").strip()
        if choice == "1":
            frames_dir = input("  Path to frames folder: ").strip()
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                image_paths += glob.glob(os.path.join(frames_dir, ext))
            image_paths = sorted(image_paths)
        else:
            vid_input  = input("  Video file paths (comma-separated): ").strip()
            video_list = [v.strip() for v in vid_input.split(",")]
            n_str      = input(f"  Frames per video (default {args.nframes}): ").strip()
            n          = int(n_str) if n_str.isdigit() else args.nframes
            frames_dir = os.path.join(output_dir, "extracted_frames")
            image_paths = extract_frames_from_videos(video_list, frames_dir, n)

    if not image_paths:
        print("[ERROR] No images found. Please check your paths.")
        sys.exit(1)

    print(f"[INFO] Ready to label {len(image_paths)} frames")
    print(f"[INFO] Output directory: {output_dir}\n")

    Labeler(
        image_paths=image_paths,
        bodyparts=bodyparts,
        scorer=args.scorer,
        output_dir=output_dir,
    ).run()


if __name__ == "__main__":
    main()
