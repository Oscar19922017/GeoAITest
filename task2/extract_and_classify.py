#!/usr/bin/env python
"""
Task 2 – Image Extraction & Classification (map / table / picture)
Author: <tu nombre>

Zero-shot con CLIP.  Si pasas --model-path, cargará un
checkpoint .pt con una cabeza de 3 clases (map, table, picture).
"""
import argparse, fitz, shutil, zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, pandas as pd
from transformers import CLIPProcessor, CLIPModel

LABELS = ["map", "table", "picture"]
MIN_SIDE = 100        # px -> descarta thumbnails

def extract_images(pdf_path, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)
    with fitz.open(pdf_path) as doc:
        for page in tqdm(doc, desc="Extract"):
            for n, info in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(doc, info[0])
                ext = ".png" if pix.alpha else ".jpg"
                fname = f"p{page.number+1:03d}_{n+1}{ext}"
                pix.save(out_dir / fname)

def clean_small(out_dir):
    for img in list(out_dir.iterdir()):
        w, h = Image.open(img).size
        if min(w, h) < MIN_SIDE:
            img.unlink()

@torch.inference_mode()
def classify_imgs(img_dir, model, processor, results_csv, copy_split):
    rows = []
    for img_path in tqdm(sorted(img_dir.iterdir()), desc="Classify"):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(text=LABELS, images=img, return_tensors="pt")
        logits = model(**inputs).logits_per_image.softmax(-1)[0]
        cls_id = int(logits.argmax())
        conf   = float(logits.max())
        rows.append((img_path.name, LABELS[cls_id], round(conf,3)))

        if copy_split:
            dst = img_dir.parent / f"imgs_{LABELS[cls_id]}"
            dst.mkdir(exist_ok=True)
            shutil.copy(img_path, dst / img_path.name)

    pd.DataFrame(rows, columns=["filename","label","confidence"])\
      .to_csv(results_csv, index=False)
    print(f"🔹 Saved {results_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True,
                    help="Input PDF (e.g. q4_2023_txg_ep.pdf)")
    ap.add_argument("--out-dir", default="imgs_raw",
                    help="Folder to store extracted images")
    ap.add_argument("--results",  default="results.csv",
                    help="CSV with filename,label,confidence")
    ap.add_argument("--model-path",
                    help="Optional .pt checkpoint with 3-class head")
    ap.add_argument("--split", action="store_true",
                    help="Copy images into imgs_map/, imgs_table/, imgs_picture/")
    ap.add_argument("--no-filter", action="store_true",
                    help="Keep thumbnails <100 px")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    extract_images(args.pdf, out_dir)
    if not args.no_filter:
        clean_small(out_dir)

    # ─── carga CLIP o tu modelo fine-tuned ─────────────────────────
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if args.model_path:
        print("Loading fine-tuned checkpoint…")
        base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        state = torch.load(args.model_path, map_location="cpu")
        base.load_state_dict(state)
        model = base.eval()
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

    classify_imgs(out_dir, model, processor, args.results, args.split)

    # (opcional) comprime resultados
    zipfile.ZipFile("imgs_raw.zip", "w", zipfile.ZIP_DEFLATED)\
           .write(out_dir, arcname=out_dir.name)

if __name__ == "__main__":
    main()
