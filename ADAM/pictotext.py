#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import easyocr
from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig
import pytesseract

def pic_to_text(image_path):
    # Load the image
    image = Image.open(image_path)

    # 1. Image Captioning with CLIP Model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    text_clip = "a photo"
    inputs_clip = clip_processor(text_clip, images=image, return_tensors="pt")
    outputs_clip = clip_model(**inputs_clip)
    description_clip = clip_processor.decode(outputs_clip["text"])

    # 2. OCR with EasyOCR
    reader = easyocr.Reader(["en"])
    results_ocr = reader.readtext(image_path)
    text_ocr = " ".join([result[1] for result in results_ocr])

    # 3. OCR with pytesseract
    text_pytesseract = pytesseract.image_to_string(image)

    # 4. Pix2Struct with HTRPipeline (Handwritten Text Recognition)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    detector_config = DetectorConfig(scale=0.4, margin=5)
    line_clustering_config = LineClusteringConfig(min_words_per_line=2)
    read_lines = read_page(img, detector_config, line_clustering_config=line_clustering_config)
    text_pix2struct = ""
    for read_line in read_lines:
        text_pix2struct += ' '.join(read_word.text for read_word in read_line) + '\n'

    # Combine the results
    combined_text = f"Image Caption (CLIP): {description_clip}\n\n" \
                    f"OCR (EasyOCR): {text_ocr}\n\n" \
                    f"OCR (pytesseract): {text_pytesseract}\n\n" \
                    f"Pix2Struct (HTRPipeline): {text_pix2struct}"

    return combined_text