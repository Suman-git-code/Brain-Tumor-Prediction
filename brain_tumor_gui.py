"""
Brain Tumor Prediction - GUI Application (PyTorch)
Run this AFTER training the model with brain_tumor_classification.py

Requirements:
    pip install torch torchvision pillow
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os

try:
    from PIL import Image, ImageTk
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "best_brain_tumor_model.pth"
IMG_SIZE    = 224
NUM_CLASSES = 2
CLASS_NAMES = ["No Tumor", "Pituitary Tumor"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────────

# ── COLORS & FONTS ──────────────────────────────────────────────────────────
BG_DARK    = "#0d1117"
BG_CARD    = "#161b22"
BG_PANEL   = "#1c2128"
ACCENT     = "#58a6ff"
RED        = "#f85149"
GREEN      = "#3fb950"
YELLOW     = "#d29922"
TEXT_MAIN  = "#e6edf3"
TEXT_SUB   = "#8b949e"
BORDER     = "#30363d"

FONT_TITLE  = ("Segoe UI", 20, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 9)
FONT_RESULT = ("Segoe UI", 16, "bold")
# ────────────────────────────────────────────────────────────────────────────


def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model


class BrainTumorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Brain Tumor Prediction")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)

        w, h = 780, 650
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self.model     = None
        self.img_path  = None
        self.photo_img = None

        if not DEPS_OK:
            tk.Label(self,
                     text="Missing libraries!\n\nRun:\npip install torch torchvision pillow",
                     bg=BG_DARK, fg=RED, font=FONT_HEAD, justify="center").pack(expand=True)
            return

        self._build_ui()
        self._load_model_async()

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=BG_CARD, pady=18)
        header.pack(fill="x")
        tk.Label(header, text="🧠  Brain Tumor Prediction",
                 bg=BG_CARD, fg=TEXT_MAIN, font=FONT_TITLE).pack()
        tk.Label(header, text="Upload an MRI scan to detect pituitary tumor",
                 bg=BG_CARD, fg=TEXT_SUB, font=FONT_SMALL).pack(pady=(2, 0))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Body
        body = tk.Frame(self, bg=BG_DARK, padx=20, pady=20)
        body.pack(fill="both", expand=True)

        # LEFT — image preview
        left = tk.Frame(body, bg=BG_CARD, width=320, height=380)
        left.pack(side="left", fill="both", expand=False)
        left.pack_propagate(False)

        tk.Label(left, text="MRI SCAN PREVIEW", bg=BG_CARD,
                 fg=TEXT_SUB, font=FONT_SMALL).pack(pady=(14, 0))

        self.canvas = tk.Canvas(left, width=280, height=280,
                                bg=BG_PANEL, highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack(pady=10, padx=20)
        self.canvas.create_text(140, 140, text="No image\nselected",
                                fill=TEXT_SUB, font=FONT_BODY,
                                justify="center", tags="placeholder")

        self.file_label = tk.Label(left, text="—", bg=BG_CARD,
                                   fg=TEXT_SUB, font=FONT_SMALL, wraplength=280)
        self.file_label.pack(pady=(0, 10))

        # RIGHT — controls + result
        right = tk.Frame(body, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(20, 0))

        self.upload_btn = tk.Button(
            right, text="📂  Choose MRI Image",
            bg=ACCENT, fg="#0d1117", font=FONT_HEAD,
            relief="flat", cursor="hand2", padx=20, pady=12,
            command=self._choose_image,
            activebackground="#79c0ff", activeforeground="#0d1117"
        )
        self.upload_btn.pack(fill="x", pady=(0, 12))

        self.predict_btn = tk.Button(
            right, text="🔍  Analyze & Predict",
            bg=BG_PANEL, fg=TEXT_SUB, font=FONT_HEAD,
            relief="flat", cursor="hand2", padx=20, pady=12,
            command=self._predict, state="disabled",
            activebackground="#2d333b", activeforeground=TEXT_MAIN
        )
        self.predict_btn.pack(fill="x")

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=12)

        # Result card
        self.result_card = tk.Frame(right, bg=BG_PANEL)
        self.result_card.pack(fill="x", pady=(0, 12))

        tk.Label(self.result_card, text="DIAGNOSIS", bg=BG_PANEL,
                 fg=TEXT_SUB, font=FONT_SMALL).pack(pady=(14, 2))

        self.result_icon = tk.Label(self.result_card, text="—",
                                    bg=BG_PANEL, fg=TEXT_SUB,
                                    font=("Segoe UI", 36))
        self.result_icon.pack()

        self.result_label = tk.Label(self.result_card, text="Awaiting analysis",
                                     bg=BG_PANEL, fg=TEXT_SUB, font=FONT_RESULT)
        self.result_label.pack()

        self.conf_label = tk.Label(self.result_card, text="",
                                   bg=BG_PANEL, fg=TEXT_SUB, font=FONT_BODY)
        self.conf_label.pack(pady=(4, 14))

        # Confidence bar
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x")
        tk.Label(right, text="CONFIDENCE", bg=BG_DARK,
                 fg=TEXT_SUB, font=FONT_SMALL).pack(anchor="w", pady=(8, 0))

        bar_bg = tk.Frame(right, bg=BORDER, height=12)
        bar_bg.pack(fill="x", pady=(4, 0))
        bar_bg.pack_propagate(False)

        self.conf_bar = tk.Frame(bar_bg, bg=ACCENT, width=0, height=12)
        self.conf_bar.place(x=0, y=0, height=12)

        self.conf_pct_label = tk.Label(right, text="0%",
                                       bg=BG_DARK, fg=TEXT_SUB, font=FONT_SMALL)
        self.conf_pct_label.pack(anchor="e")

        # Status bar
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        status_bar = tk.Frame(self, bg=BG_CARD, pady=6)
        status_bar.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="Loading model…")
        tk.Label(status_bar, textvariable=self.status_var,
                 bg=BG_CARD, fg=TEXT_SUB, font=FONT_SMALL).pack(side="left", padx=14)

        self.model_status = tk.Label(status_bar, text="⏳ Model loading",
                                     bg=BG_CARD, fg=YELLOW, font=FONT_SMALL)
        self.model_status.pack(side="right", padx=14)

    # ── MODEL LOAD ───────────────────────────────────────────────────────
    def _load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.after(0, lambda: self._set_status(
                f"'{MODEL_PATH}' not found. Run brain_tumor_classification.py first.", RED))
            return
        try:
            m = build_model()
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            self.model = m
            self.after(0, lambda: self._set_status("Model loaded — ready to predict.", GREEN, ok=True))
        except Exception as e:
            self.after(0, lambda: self._set_status(f"Model error: {e}", RED))

    def _set_status(self, msg, color=TEXT_SUB, ok=False):
        self.status_var.set(msg)
        self.model_status.config(
            text="✅ Model ready" if ok else "❌ Model error",
            fg=GREEN if ok else RED
        )

    # ── IMAGE ────────────────────────────────────────────────────────────
    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Select MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self.img_path = path
        self.file_label.config(text=os.path.basename(path), fg=TEXT_MAIN)
        self._show_preview(path)
        self._reset_result()
        if self.model:
            self.predict_btn.config(state="normal", bg=ACCENT, fg="#0d1117")

    def _show_preview(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((280, 280))
            self.photo_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            cx = 280 // 2
            cy = 280 // 2
            self.canvas.create_image(cx, cy, anchor="center", image=self.photo_img)
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not open image:\n{e}")

    def _reset_result(self):
        self.result_icon.config(text="—", fg=TEXT_SUB)
        self.result_label.config(text="Awaiting analysis", fg=TEXT_SUB)
        self.conf_label.config(text="")
        self.conf_bar.place(width=0)
        self.conf_pct_label.config(text="0%")

    # ── PREDICT ──────────────────────────────────────────────────────────
    def _predict(self):
        if not self.img_path or not self.model:
            return
        self.predict_btn.config(state="disabled", text="Analyzing…")
        self.status_var.set("Running prediction…")
        threading.Thread(target=self._run_prediction, daemon=True).start()

    def _run_prediction(self):
        try:
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            img = Image.open(self.img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = self.model(tensor)
                probs   = torch.softmax(outputs, dim=1)[0]
                pred    = torch.argmax(probs).item()
                conf    = probs[pred].item()

            self.after(0, lambda: self._show_result(pred, conf))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))
            self.after(0, lambda: self.predict_btn.config(
                state="normal", text="🔍  Analyze & Predict"))

    def _show_result(self, pred, conf):
        pct       = int(conf * 100)
        is_tumor  = pred == 1

        if is_tumor:
            icon, label, color = "⚠️", "PITUITARY TUMOR DETECTED", RED
        else:
            icon, label, color = "✅", "NO TUMOR DETECTED", GREEN

        self.result_icon.config(text=icon, fg=color)
        self.result_label.config(text=label, fg=color)
        self.conf_label.config(
            text=f"Confidence: {pct}%  |  Class: {CLASS_NAMES[pred]}",
            fg=TEXT_SUB
        )

        bar_width = self.conf_bar.master.winfo_width() or 320
        self.conf_bar.config(bg=color)
        self._animate_bar(0, pct, bar_width)

        self.predict_btn.config(state="normal", text="🔍  Analyze & Predict",
                                bg=ACCENT, fg="#0d1117")
        self.status_var.set(f"Prediction complete — {label}  ({pct}% confidence)")

    def _animate_bar(self, current, target, total_width, step=2):
        if current < target:
            self.conf_bar.place(width=int((current / 100) * total_width))
            self.conf_pct_label.config(text=f"{current}%")
            self.after(15, lambda: self._animate_bar(current + step, target, total_width))
        else:
            self.conf_bar.place(width=int((target / 100) * total_width))
            self.conf_pct_label.config(text=f"{target}%")


if __name__ == "__main__":
    app = BrainTumorApp()
    app.mainloop()
