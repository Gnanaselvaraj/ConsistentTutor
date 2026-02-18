"""
image_extractor.py: Extract images from PDFs with metadata
"""
import fitz  # PyMuPDF
from PIL import Image
import io
from typing import List, Dict, Any
import hashlib

def extract_images_from_pdf(pdf_path: str, min_width: int = 100, min_height: int = 100) -> List[Dict[str, Any]]:
    """
    Extract images from PDF with metadata.
    
    Args:
        pdf_path: Path to PDF file
        min_width: Minimum image width to extract (filters small icons)
        min_height: Minimum image height to extract
    
    Returns:
        List of dicts with keys: {image, page, bbox, source, image_hash}
    """
    doc = fitz.open(pdf_path)
    extracted_images = []
    seen_hashes = set()  # Avoid duplicate images
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Filter small images (likely icons/logos)
                if pil_image.width < min_width or pil_image.height < min_height:
                    continue
                
                # Calculate hash to detect duplicates
                image_hash = hashlib.md5(image_bytes).hexdigest()
                if image_hash in seen_hashes:
                    continue
                seen_hashes.add(image_hash)
                
                # Get image position on page
                img_rects = page.get_image_rects(xref)
                bbox = img_rects[0] if img_rects else None
                
                extracted_images.append({
                    'image': pil_image,
                    'page': page_num + 1,  # 1-indexed
                    'bbox': bbox,
                    'source': pdf_path,
                    'image_hash': image_hash,
                    'width': pil_image.width,
                    'height': pil_image.height,
                })
            
            except Exception as e:
                # Skip problematic images
                continue
    
    doc.close()
    return extracted_images
