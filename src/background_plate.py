"""Produces a person-free photo of the court by taking the per-pixel MEDIAN
across many frames spread across the video. The camera is fixed, so the
court/net/background is identical in every frame, but a given pixel is only
covered by a moving player some of the time -- across enough samples, the
median at each pixel is the real court surface, not a person.

This is deliberately not generative AI: we already have hundreds of real
frames showing every part of the court unoccluded at some point, so computing
the true background is both more accurate (no hallucinated detail) and
free/instant compared to inpainting.
"""

import argparse

import cv2
import numpy as np


def generate_background(video_path: str, num_samples: int = 40) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(num_samples):
        frame_idx = int(total * (i + 0.5) / num_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")

    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate a person-free background plate of the court.")
    parser.add_argument("--input", default="picklecam.mp4")
    parser.add_argument("--output", default="output/court_background.jpg")
    parser.add_argument("--num-samples", type=int, default=40)
    args = parser.parse_args()

    bg = generate_background(args.input, args.num_samples)
    cv2.imwrite(args.output, bg)
    print(f"Saved background plate ({args.num_samples} frame median) to {args.output}")


if __name__ == "__main__":
    main()
