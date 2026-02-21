Script for collecting images and generating VQA (Visual Question Answering) data in JSON format.

The pipeline:
- Downloads images from public sources (Pexels, Wikimedia, Wikipedia).
- Generates 4–6 English question–answer pairs per image.
- Optionally verifies QA pairs.
- Saves results to a JSON file.

Structure
vqa_fast_free_images_pipeline.py   # main script
images_3k/                         # downloaded images
vqa_3k_en.json                     # generated dataset
Requirements

Python 3.9+
pip install openai requests pillow tqdm

Set API key:
Windows PowerShell:
$env:OPENAI_API_KEY="your_key"
Pexels key:
$env:PEXELS_API_KEY="your_key"
Run
Test:
py vqa_fast_free_images_pipeline.py --target 10 --use_pexels
Full dataset:
py vqa_fast_free_images_pipeline.py --target 3000 --use_pexels --download_workers 16 --workers 8 --low_ratio 0.75 --verify
Output is a JSON file containing image paths and generated QA pairs.




LINK FOR DRIVE WITH PHOTOS: https://drive.google.com/drive/folders/1sF3ROluNaBR5FOtMs8OPGaWhAiO1qPrg?usp=sharing
