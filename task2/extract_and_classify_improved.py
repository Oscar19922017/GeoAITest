#!/usr/bin/env python
"""
Task 2 – Image Extraction & Classification (map / table / picture)
Author: <tu nombre>

Versión mejorada que extrae tanto imágenes embebidas como renderiza 
elementos de página (incluyendo tablas) como imágenes.
"""
import argparse, fitz, shutil, zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch, pandas as pd
from transformers import CLIPProcessor, CLIPModel
import cv2
import numpy as np

LABELS = ["map", "table", "picture"]
MIN_SIDE = 100        # px -> descarta thumbnails
DPI = 150            # DPI para renderizar páginas

def save_and_classify_image(pix, base_fname, out_dir, model, processor):
    """Clasifica y guarda la imagen en la carpeta correspondiente según la etiqueta"""
    from PIL import Image as PILImage
    import io
    # Convertir pixmap a PIL Image
    img_bytes = pix.tobytes("png")
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    # Clasificar
    inputs = processor(text=LABELS, images=img, return_tensors="pt")
    logits = model(**inputs).logits_per_image.softmax(-1)[0]
    cls_id = int(logits.argmax())
    label = LABELS[cls_id]
    # Guardar en carpeta correspondiente
    dst_dir = out_dir.parent / f"imgs_{label}"
    dst_dir.mkdir(exist_ok=True)
    fname = f"{base_fname}"
    img.save(dst_dir / fname)
    return fname, label, float(logits.max())

def extract_embedded_images(pdf_path, out_dir, model, processor, results):
    """Extrae imágenes embebidas del PDF y las clasifica y guarda"""
    with fitz.open(pdf_path) as doc:
        for page in tqdm(doc, desc="Extract embedded"):
            for n, info in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(doc, info[0])
                if pix.width < MIN_SIDE or pix.height < MIN_SIDE:
                    continue
                ext = ".png" if pix.alpha else ".jpg"
                base_fname = f"embedded_p{page.number+1:03d}_{n+1}{ext}"
                fname, label, conf = save_and_classify_image(pix, base_fname, out_dir, model, processor)
                results.append((fname, label, round(conf,3)))

def detect_table_regions(page_pix, min_area=5000):
    """
    Detecta posibles regiones de tablas usando detección de líneas
    """
    # Convertir a formato OpenCV
    img_data = page_pix.tobytes("ppm")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar líneas horizontales y verticales
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Detectar líneas
    horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
    
    # Combinar líneas
    table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Expandir un poco la región para capturar mejor la tabla
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2*margin)
            h = min(img.shape[0] - y, h + 2*margin)
            table_regions.append((x, y, w, h))
    
    return table_regions

def extract_page_elements(pdf_path, out_dir, model, processor, results):
    """Renderiza páginas y extrae regiones que podrían ser tablas, clasificando y guardando"""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(tqdm(doc, desc="Extract page elements")):
            # Renderizar página completa a alta resolución
            mat = fitz.Matrix(DPI/72, DPI/72)
            pix = page.get_pixmap(matrix=mat)
            
            # Detectar regiones de tablas
            try:
                table_regions = detect_table_regions(pix)
                
                # Extraer cada región detectada
                for i, (x, y, w, h) in enumerate(table_regions):
                    # Crear un clip de la región
                    clip_rect = fitz.Rect(x * 72/DPI, y * 72/DPI, 
                                        (x+w) * 72/DPI, (y+h) * 72/DPI)
                    region_pix = page.get_pixmap(matrix=mat, clip=clip_rect)
                    
                    if region_pix.width >= MIN_SIDE and region_pix.height >= MIN_SIDE:
                        base_fname = f"region_p{page_num+1:03d}_{i+1}.png"
                        fname, label, conf = save_and_classify_image(region_pix, base_fname, out_dir, model, processor)
                        results.append((fname, label, round(conf,3)))
                        
            except Exception as e:
                print(f"Warning: Could not process page {page_num+1}: {e}")
                continue

def extract_text_blocks_as_images(pdf_path, out_dir, model, processor, results):
    """Extrae bloques de texto que podrían ser tablas, clasificando y guardando"""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(tqdm(doc, desc="Extract text blocks")):
            blocks = page.get_text("dict")["blocks"]
            
            for block_num, block in enumerate(blocks):
                # Solo procesar bloques de texto
                if "lines" not in block:
                    continue
                    
                # Heurística simple: si el bloque tiene muchas líneas 
                # y caracteres tabulares, podría ser una tabla
                lines = block["lines"]
                if len(lines) >= 3:  # Al menos 3 líneas
                    text_content = ""
                    for line in lines:
                        for span in line["spans"]:
                            text_content += span["text"] + " "
                    
                    # Buscar indicadores de tabla (números, espacios, etc.)
                    tab_indicators = text_content.count('\t') + text_content.count('  ')
                    if tab_indicators > len(lines):  # Heurística simple
                        # Extraer la región del bloque
                        bbox = block["bbox"]
                        rect = fitz.Rect(bbox)
                        
                        # Expandir un poco
                        rect = rect + (-5, -5, 5, 5)
                        
                        mat = fitz.Matrix(DPI/72, DPI/72)
                        pix = page.get_pixmap(matrix=mat, clip=rect)
                        
                        if pix.width >= MIN_SIDE and pix.height >= MIN_SIDE:
                            base_fname = f"textblock_p{page_num+1:03d}_{block_num+1}.png"
                            fname, label, conf = save_and_classify_image(pix, base_fname, out_dir, model, processor)
                            results.append((fname, label, round(conf,3)))

def extract_all_images(pdf_path, out_dir, model, processor):
    """Función principal que combina todos los métodos de extracción y clasificación"""
    out_dir.mkdir(exist_ok=True, parents=True)
    
    results = []
    
    # 1. Extraer imágenes embebidas (método original)
    print("🔹 Extracting embedded images...")
    extract_embedded_images(pdf_path, out_dir, model, processor, results)
    
    # 2. Extraer regiones detectadas como tablas
    print("🔹 Extracting table regions...")
    extract_page_elements(pdf_path, out_dir, model, processor, results)
    
    # 3. Extraer bloques de texto que podrían ser tablas
    print("🔹 Extracting text blocks...")
    extract_text_blocks_as_images(pdf_path, out_dir, model, processor, results)
    
    print(f"🔹 Total extracted: {len(results)} images")
    return results

def clean_small(out_dir):
    """Elimina imágenes muy pequeñas"""
    removed = 0
    for img in list(out_dir.iterdir()):
        try:
            w, h = Image.open(img).size
            if min(w, h) < MIN_SIDE:
                img.unlink()
                removed += 1
        except Exception:
            # Si no se puede abrir, eliminar
            img.unlink()
            removed += 1
    if removed > 0:
        print(f"🔹 Removed {removed} small/invalid images")

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
    ap.add_argument("--no-filter", action="store_true",
                    help="Keep thumbnails <100 px")
    ap.add_argument("--dpi", type=int, default=150,
                    help="DPI for page rendering (default: 150)")
    args = ap.parse_args()

    global DPI
    DPI = args.dpi
    
    out_dir = Path(args.out_dir)

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

    # Extraer y clasificar todas las imágenes usando múltiples métodos
    results = extract_all_images(args.pdf, out_dir, model, processor)

    if not args.no_filter:
        clean_small(out_dir)

    # Guardar resultados en CSV
    import pandas as pd
    df = pd.DataFrame(results, columns=["filename","label","confidence"])
    df.to_csv(args.results, index=False)
    print(f"🔹 Saved {args.results}")
    
    # Mostrar estadísticas
    print("\n🔹 Classification results:")
    print(df.groupby('label').size())

    # (opcional) comprime resultados
    try:
        with zipfile.ZipFile("imgs_raw.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for file in out_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(out_dir.parent))
        print("🔹 Created imgs_raw.zip")
    except Exception as e:
        print(f"Warning: Could not create zip: {e}")

if __name__ == "__main__":
    main()