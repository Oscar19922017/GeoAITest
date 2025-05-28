#!/usr/bin/env python
"""
Task 3 – Geospatial Computer Vision
Detecta texto con coordenadas (lat/lon, DMS, UTM…) en mapas y recorta la
zona.  Salidas:
  • geocrops/segments/<img_stem>/segment_*.png   (recortes)
  • geocrops/visualizations/<img_stem>_coords.png (mapa con bbox coloreados)
  • coords.csv  (filename, n_coords, n_segments, coords_json)
"""

import re, argparse, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw
import pytesseract, pandas as pd
from tqdm import tqdm

# ── ajustes opcionales ─────────────────────────────────────────────────────────
# Descomenta en Windows si tesseract no está en %PATH%
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Patrones de coordenadas habituales
COORD_PATTERNS = [
    # DD.ddd, –DD.ddd
    r'[-+]?\d{1,3}\.\d{1,6}\s*[,;]\s*[-+]?\d{1,3}\.\d{1,6}',
    # 40°42'46"N 74°00'21"W
    r'\d{1,3}°\s*\d{1,2}[\'′]\s*\d{1,2}[\"″]?\s*[NS]\s*[,;]?\s*'
    r'\d{1,3}°\s*\d{1,2}[\'′]\s*\d{1,2}[\"″]?\s*[EW]',
    # 18T 585628 4511322  (UTM)
    r'\d{1,2}[C-X]\s+\d{5,7}\s+\d{5,7}',
    # Lat= Lon=
    r'[Ll]at[^0-9\-+]*[-+]?\d{1,3}\.\d+.*?[Ll]on[^0-9\-+]*[-+]?\d{1,3}\.\d+',
]

KEYWORDS = {'lat','latitude','lon','longitude','coord','utm','wgs','gps',
            'north','south','east','west','°'}

def preprocess(img_rgb):
    """CLAHE + adaptive threshold para mejorar OCR"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    return cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

def ocr_with_boxes(img_bin):
    """Devuelve lista de dicts con texto y bbox de cada fragmento"""
    cfg = ("--oem 3 --psm 6 "
           "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "abcdefghijklmnopqrstuvwxyz°'\".,;:=+-NSEW")
    data = pytesseract.image_to_data(img_bin, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    return [dict(text=data['text'][i],
                 x=data['left'][i], y=data['top'][i],
                 w=data['width'][i], h=data['height'][i],
                 conf=int(data['conf'][i]))
            for i in range(len(data['text'])) if int(data['conf'][i]) > 30]

def find_coord_candidates(text_blocks):
    """Detecta coordenadas por regex o keywords; devuelve lista de dicts"""
    full = ' '.join(b['text'] for b in text_blocks)
    found = []
    # regex
    for pat in COORD_PATTERNS:
        for m in re.finditer(pat, full, re.I):
            found.append({'text': m.group(), 'method':'regex', 'confidence':0.9})
    # keywords + números cercanos
    for b in text_blocks:
        tl = b['text'].lower()
        if any(k in tl for k in KEYWORDS):
            nums = re.findall(r'[-+]?\d+\.\d+', tl)
            if len(nums) >= 2:
                found.append({'text': b['text'], 'method':'keyword', 'confidence':0.5,
                              'position': b})
    return found

def bbox_from_text(text, blocks):
    words = text.split()
    for b in blocks:
        if any(w in b['text'] for w in words):
            return {k:b[k] for k in ('x','y','w','h')}
    return None

def process_image(img_path, out_root, margin=50):
    img_rgb  = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    bin_img  = preprocess(img_rgb)
    blocks   = ocr_with_boxes(bin_img)
    cands    = find_coord_candidates(blocks)

    seg_dir  = out_root/'segments'/img_path.stem
    vis_dir  = out_root/'visualizations'; vis_dir.mkdir(parents=True, exist_ok=True)
    drawn    = Image.fromarray(img_rgb)
    draw     = ImageDraw.Draw(drawn)

    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_count = 0
    for c in cands:
        pos = c.get('position') or bbox_from_text(c['text'], blocks)
        if not pos: continue
        # expand bbox
        x,y,w,h = pos['x'],pos['y'],pos['w'],pos['h']
        x, y = max(0,x-margin), max(0,y-margin)
        w, h = min(img_rgb.shape[1]-x, w+2*margin), min(img_rgb.shape[0]-y, h+2*margin)
        # save crop
        seg_img = Image.fromarray(img_rgb[y:y+h, x:x+w])
        seg_count += 1
        seg_path = seg_dir/f"segment_{seg_count}.png"
        seg_img.save(seg_path)

        # mark on visualization
        color = 'green' if c['confidence']>0.7 else 'orange'
        draw.rectangle([x,y,x+w,y+h], outline=color, width=2)
        draw.text((x, max(0,y-15)), f"{seg_count}", fill=color)

    drawn.save(vis_dir/f"{img_path.stem}_coords.png")
    return {'filename':img_path.name,
            'coordinates_found':len(cands),
            'segments_extracted':seg_count,
            'candidates':cands}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir',  default='../task2/imgs_map')
    ap.add_argument('--output-dir', default='geocrops')
    ap.add_argument('--results-file',default='coords.csv')
    args = ap.parse_args()

    in_dir  = Path(args.input_dir); out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list(in_dir.glob('*.png'))+list(in_dir.glob('*.jpg'))
    if not imgs:
        print(f"❌ No images in {in_dir}"); return

    rows = []
    for img in tqdm(imgs, desc="Maps"):
        rows.append(process_image(img, out_dir))

    pd.DataFrame(rows).to_csv(args.results_file, index=False)
    print(f"✓ Done → {args.results_file}\n  crops: {out_dir}/segments/*")

if __name__ == '__main__':
    main()
