# GeoAITest

## Automated Extraction of Data and Images from Technical Reports and Maps

### **General Description**
GeoAITest is a suite of scripts and notebooks designed to automate the extraction, classification, and structuring of relevant data from technical PDF reports, including financial presentations, images, tables, and geological maps. The goal is to facilitate large-scale analysis and digitization of information contained in complex documents, combining computer vision, natural language processing, and OCR techniques.

---

### **Main Tasks**

- **Task 1: Extraction and Consolidation of Financial and Operational Tables with AI**
  - Script/Notebook: `task1/extractor.ipynb`
  - Consolidation Notebooks:
    - `task1/consolidar_key_financial_metrics.ipynb`
    - `task1/consolidar_key_operational_metrics.ipynb`
    - `task1/consolidar_robust_cash_flow.ipynb`
  - Automates the extraction of key tables (financial and operational) from PDFs, even when they are embedded as images.
  - Utilizes OpenAI's multimodal models (GPT-4o) to identify and structure tabular information, overcoming the limitations of traditional methods.
  - Consolidates extracted tables across multiple files for unified analysis, depending on the metric of interest.
  - **Results:** Tables extracted in Excel format, ready for analysis, with error handling and robustness for scanned documents.

- **Task 2: Extraction and Classification of Images in PDFs**
  - Script: `task2/extract_and_classify_improved.py`
  - Automatically extracts all embedded images, rendered tables, and photographs present in PDFs.
  - Classifies each image into categories such as `"map"`, `"table"`, or `"picture"` using the CLIP model (OpenAI, via HuggingFace).
  - **Results:** Images saved and classified, with a summary CSV file and an option to compress results.

- **Task 3: Extraction of Geospatial and Technical Information from Maps and Blueprints**
  - Script: `task3/geocv_task3.py` (and exploratory notebook: `task3/geocv_task3_1.ipynb`)
  - Detects and extracts geographic coordinates (lat/lon, UTM, DMS) present as text in maps using OCR and regular expressions.
  - For geological blueprints without explicit coordinates, adapts the process to identify drill hole names, drilling intervals, and relevant numeric references using text patterns and regex.
  - **Results:** Image segments with coordinates or references, marked visualizations, and structured CSV files.

---

### **Key Technologies and Libraries**

- Python 3.9+
- OpenAI GPT-4o (API)
- CLIP (OpenAI, HuggingFace Transformers)
- pdfplumber, pdf2image, PyMuPDF (fitz)
- Pillow, OpenCV, pytesseract
- pandas, torch, tqdm, zipfile

---

### **General Workflow**

1. **PDF loading and analysis:** Batch processing of documents.
2. **AI-driven extraction:** Use of language and AI models to overcome the heterogeneity and visual complexity of documents.
3. **Classification and structuring:** Extracted images and data are automatically organized and classified.
4. **Export of results:** Excel tables, classified images, segments of interest, and CSV files ready for analysis and visualization.

---

### **Approach Advantages**

- **Scalable:** Designed to process large volumes of PDFs and maps automatically.
- **Flexible and adaptable:** Allows modification of prompts and extraction patterns for different types of documents.
- **AI integration:** Advanced models to maximize accuracy in contexts where classic methods fail.
- **Structured results:** Facilitates subsequent analysis and efficient digitization of information.

---

### **Execution and Requirements**

- Install the requirements with:
  ```
  pip install -r requirements.txt
  ```
- Configure your OpenAI API key and required models.
- Run each script/notebook according to the desired task:
  - **Task 1:** Run `task1/extractor.ipynb` (Jupyter Notebook) and the appropriate consolidation notebook(s):
    - `task1/consolidar_key_financial_metrics.ipynb`
    - `task1/consolidar_key_operational_metrics.ipynb`
    - `task1/consolidar_robust_cash_flow.ipynb`
  - **Task 2:** Run `python task2/extract_and_classify_improved.py --pdf <yourfile.pdf>`
  - **Task 3:** Run `python task3/geocv_task3.py --input-dir <input_folder> --output-dir <output_folder>`

---

### **Notes and Limitations**

- For geological maps with little explicit geospatial information, it is necessary to adapt and adjust extraction patterns.
- The process depends on the quality of the PDF/image and the accuracy of the OCR and AI models used.
- Developed to accelerate the digitization and analysis of information in complex technical contexts using AI and computer vision.
