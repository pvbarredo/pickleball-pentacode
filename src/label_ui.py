"""Visual naming tool: shows every candidate person photo from src.enroll in a
grid, lets you check off the ones that are the same person, type one name,
and assign it to all of them at once. Saves a mapping file that main.py's
--seed-gallery uses to start the full run with your names already attached.
"""

import argparse
import json
import os

import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox


THUMB_SIZE = (150, 200)
COLUMNS = 5


class LabelUI:
    def __init__(self, root, enroll_dir: str):
        self.root = root
        self.enroll_dir = enroll_dir
        self.root.title("Name the players")

        self.labels = sorted(
            f[:-4] for f in os.listdir(enroll_dir)
            if f.endswith(".jpg") and f != "contact_sheet.jpg"
        )
        self.assigned: dict[str, str] = {}  # placeholder -> final name (missing = discarded)
        self.check_vars: dict[str, tk.BooleanVar] = {}
        self.name_labels: dict[str, tk.Label] = {}
        self._thumb_refs = []  # keep PhotoImage references alive
        self._gif_frames: dict[str, list] = {}
        self._gif_img_labels: dict[str, tk.Label] = {}
        self._anim_tick = 0

        self._build_toolbar()
        self._build_grid()
        self._animate()

    def _build_toolbar(self):
        bar = tk.Frame(self.root, padx=10, pady=10)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(bar, text="Name for checked photo(s):").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(bar, width=20)
        self.name_entry.pack(side=tk.LEFT, padx=5)
        self.name_entry.bind("<Return>", lambda e: self.assign_name())

        tk.Button(bar, text="Assign to selected", command=self.assign_name).pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="Discard selected", command=self.discard_selected).pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="Select all", command=self.select_all).pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="Select none", command=self.select_none).pack(side=tk.LEFT, padx=5)
        tk.Button(bar, text="Done - Save", command=self.save_and_close, bg="#4CAF50", fg="white").pack(side=tk.RIGHT, padx=5)

    def _build_grid(self):
        canvas = tk.Canvas(self.root)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        self.grid_frame = tk.Frame(canvas)

        self.grid_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for i, label in enumerate(self.labels):
            row, col = divmod(i, COLUMNS)
            cell = tk.Frame(self.grid_frame, bd=1, relief=tk.GROOVE, padx=4, pady=4)
            cell.grid(row=row, column=col, padx=4, pady=4)

            gif_path = os.path.join(self.enroll_dir, f"{label}.gif")
            if os.path.exists(gif_path):
                frames = self._load_gif_frames(gif_path)
                self._gif_frames[label] = frames
                self._thumb_refs.extend(frames)
                img_label = tk.Label(cell, image=frames[0])
                img_label.pack()
                self._gif_img_labels[label] = img_label
            else:
                img_path = os.path.join(self.enroll_dir, f"{label}.jpg")
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                pil_img.thumbnail(THUMB_SIZE)
                tk_img = ImageTk.PhotoImage(pil_img)
                self._thumb_refs.append(tk_img)
                tk.Label(cell, image=tk_img).pack()

            tk.Label(cell, text=label, font=("Segoe UI", 9, "bold")).pack()

            var = tk.BooleanVar(value=False)
            self.check_vars[label] = var
            tk.Checkbutton(cell, text="select", variable=var).pack()

            name_lbl = tk.Label(cell, text="(unnamed)", fg="gray")
            name_lbl.pack()
            self.name_labels[label] = name_lbl

    @staticmethod
    def _load_gif_frames(gif_path: str) -> list:
        pil_img = Image.open(gif_path)
        frames = []
        try:
            while True:
                frame = pil_img.copy().convert("RGB")
                frame.thumbnail(THUMB_SIZE)
                frames.append(ImageTk.PhotoImage(frame))
                pil_img.seek(pil_img.tell() + 1)
        except EOFError:
            pass
        return frames or [ImageTk.PhotoImage(Image.new("RGB", THUMB_SIZE))]

    def _animate(self):
        self._anim_tick += 1
        for label, frames in self._gif_frames.items():
            self._gif_img_labels[label].configure(image=frames[self._anim_tick % len(frames)])
        self.root.after(80, self._animate)

    def _checked_labels(self):
        return [label for label, var in self.check_vars.items() if var.get()]

    def assign_name(self):
        name = self.name_entry.get().strip()
        checked = self._checked_labels()
        if not name or not checked:
            messagebox.showinfo("Name the players", "Type a name and check at least one photo first.")
            return
        for label in checked:
            self.assigned[label] = name
            self.name_labels[label].config(text=f"-> {name}", fg="#2a7a2a")
            self.check_vars[label].set(False)
        self.name_entry.delete(0, tk.END)

    def discard_selected(self):
        for label in self._checked_labels():
            self.assigned.pop(label, None)
            self.name_labels[label].config(text="(discarded)", fg="#b00000")
            self.check_vars[label].set(False)

    def select_all(self):
        for var in self.check_vars.values():
            var.set(True)

    def select_none(self):
        for var in self.check_vars.values():
            var.set(False)

    def save_and_close(self):
        unnamed = [
            label for label in self.labels
            if label not in self.assigned and self.name_labels[label].cget("text") != "(discarded)"
        ]
        if unnamed and not messagebox.askyesno(
            "Unnamed photos",
            f"{len(unnamed)} photo(s) have no name and will be discarded: {', '.join(unnamed)}.\nContinue?",
        ):
            return

        out_path = os.path.join(self.enroll_dir, "name_mapping.json")
        with open(out_path, "w") as f:
            json.dump(self.assigned, f, indent=2)
        messagebox.showinfo("Saved", f"Saved {len(self.assigned)} name assignment(s) to {out_path}")
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enroll-dir", default="output/enroll")
    args = parser.parse_args()

    root = tk.Tk()
    LabelUI(root, args.enroll_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
