# Step 1: Download dataset
python download_dataset.py --preset small --output-dir data/MIMIC-CXR-RRG_small

# Step 2: Run Phase I pipeline (creates entities_with_cui.pkl and mention2cui.pkl)
python main.py `
  --dataset-dir data/MIMIC-CXR-RRG_small `
  --mrconso data/umls-2025AB-mrconso/2025AB/META/MRCONSO.RRF `
  --output-dir outputs/KG

# Step 3: Run Phase II demo
python -m src.models.main --backbone biomedclip --batch-size 2