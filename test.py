# import argparse
# import torch
# import cv2
# import numpy as np
# from model import AODnet   # ⚠️ must match model.py

# # ---------------- ARGUMENTS ----------------
# parser = argparse.ArgumentParser(description="AOD-Net Dehazing Inference")

# parser.add_argument("--source", type=str, required=True,
#                     choices=["image", "video", "webcam", "rtsp"],
#                     help="Input source type")

# parser.add_argument("--image", type=str, help="Path to input image")
# parser.add_argument("--video", type=str, help="Path to input video")
# parser.add_argument("--rtsp", type=str, help="RTSP stream URL")
# parser.add_argument("--webcam_id", type=int, default=0, help="Webcam device ID")

# parser.add_argument("--model", type=str, default=r".\model\nets\AOD_9.pkl",
#                     help="Path to trained AOD-Net checkpoint")

# parser.add_argument("--output", type=str, default="output_dehazed.mp4",
#                     help="Output video file")

# args = parser.parse_args()

# # ---------------- LOAD MODEL ----------------
# ckpt = torch.load(args.model, map_location="cpu")

# model = AODnet()
# model.load_state_dict(ckpt["state_dict"])   # checkpoint-safe
# model.eval()

# # ---------------- PRE / POST ----------------
# def preprocess(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32) / 255.0
#     img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
#     return img

# def postprocess(tensor):
#     img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     img = (img * 255).clip(0, 255).astype(np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     return img

# # ---------------- IMAGE ----------------
# if args.source == "image":
#     if args.image is None:
#         raise ValueError("❌ --image path required for image source")

#     frame = cv2.imread(args.image)
#     with torch.no_grad():
#         out = model(preprocess(frame))
#     result = postprocess(out)

#     cv2.imwrite("output.png", result)
#     print("✅ Image saved as output.png")

# # ---------------- VIDEO / WEBCAM / RTSP ----------------
# else:
#     if args.source == "video":
#         cap = cv2.VideoCapture(args.video)
#     elif args.source == "webcam":
#         cap = cv2.VideoCapture(args.webcam_id)
#     elif args.source == "rtsp":
#         cap = cv2.VideoCapture(args.rtsp)

#     if not cap.isOpened():
#         raise RuntimeError("❌ Failed to open video source")

#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps    = cap.get(cv2.CAP_PROP_FPS) or 25

#     writer = cv2.VideoWriter(
#         args.output,
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height)
#     )

#     print("▶️ Processing... Press Q to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         with torch.no_grad():
#             out = model(preprocess(frame))
#         result = postprocess(out)

#         writer.write(result)
#         cv2.imshow("AOD-Net Dehazing", result)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     writer.release()
#     cv2.destroyAllWindows()

#     print("✅ Output saved to", args.output)

#Neeche wala comparision wala plus option 

import argparse
import os
import torch
import cv2
import numpy as np
from model import AODnet   # must match model.py

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser(description="AOD-Net Flexible Inference")

parser.add_argument("--source", type=str, required=True,
                    choices=["video", "webcam", "rtsp"],
                    help="Input source type")

parser.add_argument("--video", type=str, help="Path to input video")
parser.add_argument("--rtsp", type=str, help="RTSP stream URL")
parser.add_argument("--webcam_id", type=int, default=0)

parser.add_argument("--model", type=str, default=r".\model\nets\backup.pkl")
parser.add_argument("--outdir", type=str, default="outputs")
parser.add_argument("--outfile", type=str, default="dehazed.mp4")

parser.add_argument("--compare", action="store_true",
                    help="Show side-by-side comparison (Before | After)")

args = parser.parse_args()

# ---------------- SETUP ----------------
os.makedirs(args.outdir, exist_ok=True)
out_path = os.path.join(args.outdir, args.outfile)

# ---------------- LOAD MODEL ----------------
ckpt = torch.load(args.model, map_location="cpu")
model = AODnet()
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ---------------- PRE / POST ----------------
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def postprocess(tensor):
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# ---------------- OPEN SOURCE ----------------
if args.source == "video":
    cap = cv2.VideoCapture(args.video)
elif args.source == "webcam":
    cap = cv2.VideoCapture(args.webcam_id)
elif args.source == "rtsp":
    cap = cv2.VideoCapture(args.rtsp)

if not cap.isOpened():
    raise RuntimeError("❌ Could not open input source")

# ---------------- VIDEO WRITER ----------------
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25

out_size = (w * 2, h) if args.compare else (w, h)

writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    out_size
)

mode = "COMPARISON" if args.compare else "OUTPUT ONLY"
print(f"▶️ Processing ({mode})... Press Q to quit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    with torch.no_grad():
        out = model(preprocess(frame))
    dehazed = postprocess(out)

    dehazed = cv2.resize(dehazed, (w, h))

    if args.compare:
        combined = np.hstack((frame, dehazed))
        cv2.putText(combined, "Before", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, "After (AOD-Net)", (w + 30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        display = combined
    else:
        display = dehazed

    writer.write(display)
    cv2.imshow("AOD-Net Output", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"✅ Saved output to {out_path}")
