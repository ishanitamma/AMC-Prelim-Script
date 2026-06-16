# """
# Arthrogryposis ROM — Frame Labeling Tool
# =========================================
# Standalone OpenCV-based labeling tool.
# Saves output in DeepLabCut-compatible CSV + H5 format.

# Dependencies (all lightweight, no DLC GUI required):
#     pip install opencv-python numpy pandas pyyaml h5py

# Usage:
#     python labeler.py --frames "C:\path\to\files\name" --scorer "Write your name here" --output "C:\path\to\output"
# """

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
# DEFAULT KEYPOINTS  (edit here or pass via config.yaml)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BODYPARTS = [
    "nose",
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
    "left_middlemcp",
    "mid_spine",
   
]

# Colour per keypoint (BGR)
COLORS = [
    (0,   255, 0),    # nose  — green
    (0,   200, 255),  # right_shoulder     — yellow-ish
    (0,   100, 255),  # right_elbow     — orange
    (255, 100, 0),    # right_thumbmcp   — blue
    (255, 200, 0),    # right_thumbtip     — cyan
    (255, 50,  50),   # right_middlemcp      — light blue
    (0,   0,   255),  # left_shoulder       — red
    (180, 0,   255),  # left_elbow       — purple
    (0,   255, 200),  # left_wrist       — teal
    (50,  255, 150),  # left_thumbmcp       — mint
    (255, 255, 255),  # left_thumbtip          — white
    (128, 0,   128),  # left_indexmcp    — dark purple
    (0,   255, 255),  # left_indextip    — yellow
    (128, 255, 0),    # left_middlemcp   — lime green
    (255, 0,   255),  # mid_spine        — magenta
]

DOT_RADIUS   = 6
FONT         = cv2.FONT_HERSHEY_SIMPLEX
PANEL_WIDTH  = 260   # sidebar width in pixels


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
        y = 56 + i * 30
        color  = clamp_color(i)
        labeled = labels.get(bp) is not None

        # Highlight current
        if i == current_bp_idx:
            cv2.rectangle(canvas,
                          (canvas.shape[1] - PANEL_WIDTH + 4, y - 14),
                          (canvas.shape[1] - 4, y + 10),
                          (60, 60, 100), -1)

        # Dot indicator
        cv2.circle(canvas,
                   (canvas.shape[1] - PANEL_WIDTH + 18, y - 2),
                   5, color, -1 if labeled else 1)

        # Label
        status = "✓" if labeled else " "
        text   = f"{status} {bp}"
        cv2.putText(canvas, text,
                    (canvas.shape[1] - PANEL_WIDTH + 30, y + 4),
                    FONT, 0.42,
                    (220, 220, 220) if i == current_bp_idx else (160, 160, 160),
                    1, cv2.LINE_AA)

    # Controls help
    help_y = img_h - 200
    cv2.line(canvas,
             (canvas.shape[1] - PANEL_WIDTH + 8, help_y - 10),
             (canvas.shape[1] - 8, help_y - 10),
             (80, 80, 80), 1)
    tips = [
        "CONTROLS",
        "LClick  : place point",
        "Del/r   : remove point",   # ← updated: keyboard removal
        "Tab/n   : next keypoint",
        "Shift+Tab: prev keypoint",
        "d / ->  : next frame",
        "a / <-  : prev frame",
        "s       : save progress",
        "h       : skip frame",
        "q / Esc : quit & save",
    ]
    for j, tip in enumerate(tips):
        cv2.putText(canvas, tip,
                    (canvas.shape[1] - PANEL_WIDTH + 10, help_y + j * 18),
                    FONT, 0.36,
                    (200, 200, 200) if j == 0 else (130, 130, 130),
                    1, cv2.LINE_AA)


def draw_labels_on_image(img, labels, bodyparts, current_bp_idx, zoom_factor=1.0):
    """Draw all placed keypoint dots and their names onto the image."""
    out = img.copy()
    for i, bp in enumerate(bodyparts):
        pt = labels.get(bp)
        if pt is None:
            continue
        x, y   = int(pt[0]), int(pt[1])
        color   = clamp_color(i)
        radius  = max(3, int(DOT_RADIUS * zoom_factor))
        thick   = 2 if i == current_bp_idx else 1

        cv2.circle(out, (x, y), radius, color, thick)
        cv2.circle(out, (x, y), 2, (255, 255, 255), -1)

        # Label text offset so it doesn't cover the dot
        tx, ty = x + radius + 3, y - radius
        cv2.putText(out, bp, (tx + 1, ty + 1), FONT, 0.38, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(out, bp, (tx,     ty),     FONT, 0.38, color,         1, cv2.LINE_AA)
    return out


def draw_status_bar(canvas, frame_idx, total_frames, n_labeled, n_total_bp,
                    img_path, current_bp):
    """Draw a status bar at the top of the window."""
    bar_h = 38
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (20, 20, 40), -1)

    fname = Path(img_path).name
    pct   = int(100 * n_labeled / n_total_bp) if n_total_bp else 0
    bar_text = (f"Frame {frame_idx + 1}/{total_frames}  |  "
                f"{fname}  |  "
                f"Labeled: {n_labeled}/{n_total_bp} ({pct}%)  |  "
                f"Active: {current_bp}")

    cv2.putText(canvas, bar_text, (10, 25),
                FONT, 0.45, (200, 220, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_json(all_labels, save_path):
    """Save intermediate progress as JSON (always safe)."""
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
    Output: CollectedData_<scorer>.csv
    This format is directly loadable by DLC's training pipeline.
    """
    rows = []
    img_paths = sorted(all_labels.keys())

    for img_path in img_paths:
        frame_labels = all_labels[img_path]
        row = {}
        for bp in bodyparts:
            pt = frame_labels.get(bp)
            if pt is not None:
                row[(scorer, bp, "x")] = pt[0]
                row[(scorer, bp, "y")] = pt[1]
            else:
                row[(scorer, bp, "x")] = np.nan
                row[(scorer, bp, "y")] = np.nan
        rows.append((img_path, row))

    index      = pd.Index([r[0] for r in rows], name="image")
    col_tuples = [(scorer, bp, coord)
                  for bp in bodyparts for coord in ["x", "y"]]
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
    print(f"\n[SAVED] CSV  → {csv_path}")

    # Also save H5 for DLC compatibility
    try:
        h5_path = os.path.join(output_dir, f"CollectedData_{scorer}.h5")
        df.to_hdf(h5_path, key="df_with_missing", mode="w")
        print(f"[SAVED] H5   → {h5_path}")
    except Exception as e:
        print(f"[WARN]  H5 save failed (h5py not installed?): {e}")
        print("        CSV export is sufficient for DLC training.")

    return df


def export_summary(all_labels, bodyparts, output_dir):
    """Print and save a labeling summary."""
    total_frames   = len(all_labels)
    labeled_counts = {bp: 0 for bp in bodyparts}
    for frame_labels in all_labels.values():
        for bp in bodyparts:
            if frame_labels.get(bp) is not None:
                labeled_counts[bp] += 1

    print("\n" + "="*55)
    print("  LABELING SUMMARY")
    print("="*55)
    print(f"  Total frames labeled: {total_frames}")
    print(f"  {'Keypoint':<22}  {'Labeled':>7}  {'Coverage':>9}")
    print("  " + "-"*43)
    for bp, count in labeled_counts.items():
        pct = 100 * count / total_frames if total_frames else 0
        print(f"  {bp:<22}  {count:>7}  {pct:>8.1f}%")
    print("="*55)

    summary = {
        "total_frames": total_frames,
        "keypoint_coverage": labeled_counts,
        "exported_at": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "labeling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LABELING LOOP
# ─────────────────────────────────────────────────────────────────────────────

class Labeler:
    def __init__(self, image_paths, bodyparts, scorer, output_dir,
                 existing_labels=None):
        self.image_paths  = image_paths
        self.bodyparts    = bodyparts
        self.scorer       = scorer
        self.output_dir   = output_dir
        self.json_path    = os.path.join(output_dir, "labels_progress.json")

        # Load existing labels or start fresh
        self.all_labels   = existing_labels or load_json(self.json_path)

        self.frame_idx    = 0
        self.bp_idx       = 0
        self.zoom         = 1.0
        self.click_pos    = None

        # Find first unlabeled frame to resume from
        for i, p in enumerate(image_paths):
            if p not in self.all_labels:
                self.frame_idx = i
                break

    # ── mouse callback ──────────────────────────────────────────────────────

    def mouse_callback(self, event, x, y, flags, param):
        img_w = param["img_w"]
        img_h_display = param["img_h_display"]

        # Ignore clicks on sidebar
        if x >= img_w:
            return
        # Ignore clicks on status bar
        if y < 38:
            return

        actual_y = y - 38  # offset for status bar

        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coords → original image coords
            ox = int(x / self.zoom)
            oy = int(actual_y / self.zoom)
            bp = self.bodyparts[self.bp_idx]
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
            else:
                # All labeled — stay on current
                pass

        # Right-click removed; use Del / Backspace / r to remove the active point.

    # ── render ──────────────────────────────────────────────────────────────

    def render(self, img_orig):
        """Build the complete display frame."""
        frame_path = self.image_paths[self.frame_idx]
        labels     = self.all_labels.get(frame_path, {})

        # Scale image
        h, w = img_orig.shape[:2]
        disp_w = int(w * self.zoom)
        disp_h = int(h * self.zoom)
        img_disp = cv2.resize(img_orig, (disp_w, disp_h))

        # Draw keypoints on scaled image
        img_disp = draw_labels_on_image(
            img_disp, labels, self.bodyparts, self.bp_idx, self.zoom)

        # Status bar
        status_bar = np.zeros((38, disp_w + PANEL_WIDTH, 3), dtype=np.uint8)
        n_labeled  = sum(1 for v in labels.values() if v is not None)

        # Compose: [status bar] on top, [image | sidebar] below
        sidebar_h = max(disp_h, len(self.bodyparts) * 30 + 250)
        canvas    = np.zeros((38 + sidebar_h, disp_w + PANEL_WIDTH, 3), dtype=np.uint8)
        canvas[:38, :, :]                     = status_bar
        canvas[38:38 + disp_h, :disp_w, :]   = img_disp

        draw_status_bar(canvas, self.frame_idx, len(self.image_paths),
                        n_labeled, len(self.bodyparts),
                        frame_path, self.bodyparts[self.bp_idx])
        draw_sidebar(canvas, self.bodyparts, self.bp_idx,
                     labels, sidebar_h)

        return canvas, w, disp_h

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self):
        win_name = "Arthrogryposis ROM Labeler  —  press 'q' to quit & save"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        print("\n" + "="*55)
        print("  LABELING TOOL STARTED")
        print("="*55)
        print(f"  Frames   : {len(self.image_paths)}")
        print(f"  Keypoints: {len(self.bodyparts)}")
        print(f"  Scorer   : {self.scorer}")
        print(f"  Output   : {self.output_dir}")
        print("="*55)
        print("  Controls: LClick=place  Del/r=remove  Tab=next BP")
        print("            d/→=next frame  a/←=prev frame")
        print("            s=save  h=skip frame  q=quit & save")
        print("="*55 + "\n")

        while True:
            img_orig  = cv2.imread(self.image_paths[self.frame_idx])
            if img_orig is None:
                print(f"[WARN] Could not read: {self.image_paths[self.frame_idx]}")
                self.frame_idx = min(self.frame_idx + 1, len(self.image_paths) - 1)
                continue

            canvas, img_w, img_h_display = self.render(img_orig)

            # Update mouse callback with current image dimensions
            cv2.setMouseCallback(win_name, self.mouse_callback,
                                 {"img_w": img_w, "img_h_display": img_h_display})
            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(30) & 0xFF

            # ── Quit ──────────────────────────────────────────────────────
            if key in (ord('q'), 27):  # q or Esc
                print("\n[INFO] Saving and quitting...")
                break

            # ── Next frame ────────────────────────────────────────────────
            elif key in (ord('d'), 83):  # d or →
                self.frame_idx = min(self.frame_idx + 1, len(self.image_paths) - 1)
                self.bp_idx    = 0

            # ── Prev frame ────────────────────────────────────────────────
            elif key in (ord('a'), 81):  # a or ←
                self.frame_idx = max(self.frame_idx - 1, 0)
                self.bp_idx    = 0

            # ── Next keypoint ─────────────────────────────────────────────
            elif key in (ord('\t'), ord('n')):  # Tab or n
                self.bp_idx = (self.bp_idx + 1) % len(self.bodyparts)

            # ── Prev keypoint ─────────────────────────────────────────────
            elif key == ord('N'):  # Shift+Tab equivalent
                self.bp_idx = (self.bp_idx - 1) % len(self.bodyparts)

            # ── Select keypoint by number ─────────────────────────────────
            elif ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if idx < len(self.bodyparts):
                    self.bp_idx = idx

            # ── Remove active keypoint ────────────────────────────────────
            elif key in (127, 8, ord('r')):  # Del, Backspace, or r
                bp         = self.bodyparts[self.bp_idx]
                frame_path = self.image_paths[self.frame_idx]
                if frame_path in self.all_labels:
                    self.all_labels[frame_path].pop(bp, None)

            # ── Skip frame (mark as intentionally skipped) ────────────────
            elif key == ord('h'):
                fp = self.image_paths[self.frame_idx]
                if fp not in self.all_labels:
                    self.all_labels[fp] = {}
                self.all_labels[fp]["__skipped__"] = True
                self.frame_idx = min(self.frame_idx + 1, len(self.image_paths) - 1)

            # ── Save progress ─────────────────────────────────────────────
            elif key == ord('s'):
                save_json(self.all_labels, self.json_path)
                print(f"[SAVED] Progress → {self.json_path}")

            # ── Zoom in/out ───────────────────────────────────────────────
            elif key == ord('+') or key == ord('='):
                self.zoom = min(self.zoom + 0.1, 3.0)
            elif key == ord('-'):
                self.zoom = max(self.zoom - 0.1, 0.3)

            # ── Jump to frame ─────────────────────────────────────────────
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

        # Save JSON progress
        save_json(self.all_labels, self.json_path)

        # Filter out skipped frames before exporting
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
    """
    Extract evenly-spaced frames from a list of video files.
    Returns list of saved image paths.
    """
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
            fname = f"{vname}_frame{idx:05d}.png"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame)
            saved_paths.append(fpath)

        cap.release()

    print(f"[INFO] {len(saved_paths)} frames extracted to {output_dir}")
    return saved_paths


def load_bodyparts_from_dlc_config(config_path):
    """Load bodyparts list from a DLC config.yaml file."""
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
    parser.add_argument("--frames",   type=str, default=None,
                        help="Path to folder of extracted PNG/JPG frames")
    parser.add_argument("--videos",   type=str, default=None,
                        help="Comma-separated paths to video files")
    parser.add_argument("--project",  type=str, default=None,
                        help="Path to DLC project root (reads config.yaml automatically)")
    parser.add_argument("--scorer",   type=str, default="Researcher",
                        help="Your name/scorer ID (default: Researcher)")
    parser.add_argument("--output",   type=str, default="./labeled_data",
                        help="Output directory for labels (default: ./labeled_data)")
    parser.add_argument("--nframes",  type=int, default=30,
                        help="Frames to extract per video (default: 30)")
    parser.add_argument("--bodyparts",type=str, default=None,
                        help="Comma-separated list of body parts (overrides defaults)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  ARTHROGRYPOSIS ROM — LABELING TOOL")
    print("="*55)

    # ── Determine bodyparts ──────────────────────────────────────────────────
    if args.bodyparts:
        bodyparts = [b.strip() for b in args.bodyparts.split(",")]
    elif args.project:
        config_path = os.path.join(args.project, "config.yaml")
        if os.path.exists(config_path):
            bodyparts = load_bodyparts_from_dlc_config(config_path)
            print(f"[INFO] Loaded {len(bodyparts)} keypoints from {config_path}")
        else:
            bodyparts = DEFAULT_BODYPARTS
    else:
        bodyparts = DEFAULT_BODYPARTS

    print(f"[INFO] Keypoints ({len(bodyparts)}): {', '.join(bodyparts)}")

    # ── Determine output dir ─────────────────────────────────────────────────
    output_dir = args.output
    if args.project:
        output_dir = os.path.join(args.project, "labeled_data", "labeling_session")
    os.makedirs(output_dir, exist_ok=True)

    # ── Get image paths ──────────────────────────────────────────────────────
    image_paths = []

    if args.frames:
        # Load from existing frames folder
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        for ext in exts:
            image_paths += glob.glob(os.path.join(args.frames, ext))
        image_paths = sorted(image_paths)
        print(f"[INFO] Found {len(image_paths)} frames in {args.frames}")

    elif args.videos:
        # Extract frames from videos first
        video_list = [v.strip() for v in args.videos.split(",")]
        frames_dir = os.path.join(output_dir, "extracted_frames")
        image_paths = extract_frames_from_videos(
            video_list, frames_dir, args.nframes)

    elif args.project:
        # Auto-find frames in DLC project structure
        frames_dir = os.path.join(args.project, "labeled-data")
        for ext in ("*.png", "*.jpg"):
            image_paths += glob.glob(
                os.path.join(frames_dir, "**", ext), recursive=True)
        image_paths = sorted(image_paths)
        if not image_paths:
            # Try extracting from project videos
            videos_dir = os.path.join(args.project, "videos")
            if os.path.exists(videos_dir):
                vids = (glob.glob(os.path.join(videos_dir, "*.mp4")) +
                        glob.glob(os.path.join(videos_dir, "*.avi")) +
                        glob.glob(os.path.join(videos_dir, "*.mov")))
                frames_dir = os.path.join(output_dir, "extracted_frames")
                image_paths = extract_frames_from_videos(
                    vids, frames_dir, args.nframes)
        print(f"[INFO] Found {len(image_paths)} frames in project")

    else:
        # Interactive: ask for input
        print("\n  No input specified. Running in interactive mode.")
        print("  Options:")
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

    # ── Start labeler ────────────────────────────────────────────────────────
    labeler = Labeler(
        image_paths  = image_paths,
        bodyparts    = bodyparts,
        scorer       = args.scorer,
        output_dir   = output_dir,
    )
    labeler.run()


if __name__ == "__main__":
    main()