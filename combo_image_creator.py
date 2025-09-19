#!/usr/bin/env python3
"""
Combo Image Creator
Uses product descriptions to generate eye-catching combo images using DALL-E 3.
"""

import json
import os
import base64
import requests
import tempfile
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComboImageCreator:
    """Creates combo images using DALL-E 3 with product descriptions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the combo image creator."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass as parameter.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize S3 client
        self.s3_bucket = os.getenv("S3_BUCKET_NAME")
        if not self.s3_bucket:
            raise ValueError("S3_BUCKET_NAME environment variable is required")
        
        self.s3_client = boto3.client('s3')
    
    def upload_to_s3(self, local_file_path: str, s3_key: str) -> str:
        """
        Upload a file to S3 and return the public URL.
        
        Args:
            local_file_path: Path to the local file to upload
            s3_key: S3 key (path) where the file should be stored
            
        Returns:
            Public S3 URL of the uploaded file
        """
        try:
            logger.info(f"Uploading {local_file_path} to S3: s3://{self.s3_bucket}/{s3_key}")
            
            # Upload file to S3
            self.s3_client.upload_file(
                local_file_path,
                self.s3_bucket,
                s3_key,
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # Generate public URL
            s3_url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            logger.info(f"Successfully uploaded to S3: {s3_url}")
            
            return s3_url
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {str(e)}")
            raise
    
    def download_image_from_s3_url(self, url: str, temp_dir: Path) -> Path:
        """
        Download an image from an S3 URL and save it to a temporary file.
        
        Args:
            url: The S3 URL to download from
            temp_dir: Directory to save the temporary file
            
        Returns:
            Path to the downloaded image file
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Get file extension from URL or content type
            if url.lower().endswith(('.png', '.jpg', '.jpeg')):
                ext = url.split('.')[-1].lower()
            else:
                # Try to determine from content type
                content_type = response.headers.get('content-type', '')
                if 'png' in content_type:
                    ext = 'png'
                elif 'jpeg' in content_type or 'jpg' in content_type:
                    ext = 'jpg'
                else:
                    ext = 'png'  # default fallback
            
            # Create a temporary file with the correct extension
            temp_file = temp_dir / f"temp_image_{hash(url)}.{ext}"
            
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded image from S3: {url} -> {temp_file.name}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            raise

    def download_and_save_image(self, image_url: str, combo_name: str, combo_id: str) -> str:
        """Download generated image and save it locally."""
        try:
            # Create combo_images directory if it doesn't exist
            os.makedirs('combo_images', exist_ok=True)
            
            # Clean combo name for filename
            safe_name = "".join(c for c in combo_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_name = safe_name.replace(' ', '_').lower()
            
            # Create filename
            filename = f"{combo_id}_{safe_name}.png"
            file_path = os.path.join('combo_images', filename)
            
            # Download image
            logger.info(f"Downloading generated image to: {file_path}")
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Save image
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully saved image to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to download and save image: {str(e)}")
            return ""
    
    def create_combo_image_prompt(self, combo: Dict[str, Any]) -> str:
        """Create a prompt for generating combo images."""
        return """Create a meticulously arranged, high-resolution flat-lay collage of the provided fashion items. The layout should be clean, symmetrical, and visually balanced, emulating a premium fashion catalog aesthetic. Each item must be clearly visible, evenly spaced, and precisely aligned on a neutral, light-colored background. Incorporate soft, natural shadows to add depth without distracting from the items. Ensure all items are scaled proportionally to create a harmonious and aesthetically pleasing outfit presentation, suitable for a professional lookbook."""
    
    def analyze_product_images(self, combo: Dict[str, Any]) -> str:
        """
        Analyze product images to create a detailed description for combo image generation.
        
        Args:
            combo: Combo data containing products with S3 URLs
            
        Returns:
            Detailed product descriptions
        """
        try:
            products = combo.get('products', [])
            product_descriptions = []
            
            for product in products:
                product_descriptions.append(f"{product.get('role', 'item').title()}: Product {product.get('product_id', '')} - {product.get('styling_notes', '')}")
            
            return "\n".join(product_descriptions)
            
        except Exception as e:
            logger.error(f"Failed to analyze product images: {str(e)}")
            return "Product analysis failed"
    
    def generate_combo_image(self, combo: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a combo image using DALL-E 3 with product images from S3 URLs.
        
        Args:
            combo: Combo data containing products with S3 URLs and metadata
            
        Returns:
            Dictionary with image generation results
        """
        try:
            combo_name = combo.get('combo_name', 'Fashion Combo')
            combo_id = combo.get('combo_id', 'unknown')
            products = combo.get('products', [])
            
            logger.info(f"Generating combo image for: {combo_name}")
            
            # Create temporary directory for downloaded images
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Download all product images from S3 URLs
                image_files = []
                for product in products:
                    if 's3_url' in product:
                        try:
                            downloaded_path = self.download_image_from_s3_url(product['s3_url'], temp_path)
                            image_files.append(downloaded_path)
                        except Exception as e:
                            logger.error(f"Failed to download {product['s3_url']}: {e}")
                            continue
                
                if not image_files:
                    raise ValueError("No images were successfully downloaded from S3 URLs")
                
                logger.info(f"Successfully downloaded {len(image_files)} images from S3")
                
                # Analyze product images to get detailed descriptions
                product_descriptions = self.analyze_product_images(combo)
                
                # Create enhanced prompt with product details
                enhanced_prompt = """
"**CRITICAL DIRECTIVE: Generate a SINGLE, HIGHLY ARTISTIC and CAPTIVATING LANDSCAPE flat-lay collage of ALL UNIQUE FASHION ITEMS PROVIDED.** The final image must be an absolutely stunning, mesmerizing, and visually irresistible presentation, meticulously crafted for a premium email campaign.

**LANDSCAPE ORIENTATION REQUIREMENT:**
* **MANDATORY LANDSCAPE FORMAT:** Create a wide, horizontal composition that works perfectly for email layouts and social media sharing.
* **OPTIMAL DIMENSIONS:** Design for a 16:9 or 3:2 aspect ratio that looks professional in email templates.
* **HORIZONTAL ARRANGEMENT:** Arrange all items in a flowing, horizontal layout that guides the eye from left to right.

**ABSOLUTE ITEM INTEGRITY (NO EXCEPTIONS):**
* **DO NOT REPEAT ANY ITEM WHATSOEVER.** Each unique input product appears ONLY ONCE in the collage.
* **EVERY SINGLE ITEM MUST BE 100% VISIBLE, UNCROPPED, AND UNMUTILATED.** Ensure full edges of ALL items are perfectly within the frame and completely intact.
* **NO OVERLAPPING OF ITEMS.** Products must be distinctly separated and not cover each other in any way.

**EXCEPTIONAL VISUAL AESTHETIC & DYNAMIC BACKGROUND:**
The background is paramount to the appeal. Instead of a plain or simple neutral, **create a sophisticated, artistic, and visually engaging background that dynamically complements the style and color palette of the specific products.**
* **CONSIDER AESTHETIC ALIGNMENT:** Imagine a background that feels like a natural extension of a high-fashion photoshoot – perhaps a sun-drenched, textured surface (like fine sand, a weathered natural wood, a muted stone, or a luxurious linen), or a subtly stylized gradient that evokes a sense of travel, summer, or relaxed elegance. It should be visually rich but not distracting, adding significant 'wow' factor.
* **COLOR HARMONY:** The background's colors should enhance and make the products pop, creating a cohesive and stunning visual narrative.
* **LIGHTING & SHADOWS:** Employ exquisite, soft, and natural lighting with artfully cast shadows to impart deep dimension and a palpable sense of quality and luxury.

**MASTERFUL LANDSCAPE ARRANGEMENT & PRECISION:**
* **HORIZONTAL FLOW:** Create a natural left-to-right flow that tells a visual story.
* **IMPECCABLE SYMMETRY & BALANCE:** Achieve a flawless, symmetrical, and perfectly balanced composition that feels inherently elegant and harmonious.
* **PRECISION ALIGNMENT:** All items must be precisely aligned in a horizontal arrangement with absolute perfection.
* **CONSISTENT SPACING:** Maintain perfectly even and aesthetically pleasing spacing between all elements to prevent any clutter and highlight each product's form.
* **PROPORTIONAL SCALING:** Scale all items intelligently and proportionally to ensure the entire collage appears cohesive, natural, and high-impact in landscape format.

The ultimate goal is to generate a LANDSCAPE image so flawlessly stunning and artistically composed that it immediately captivates the viewer, instilling an urgent desire to click and explore, making it the undeniable star of any email campaign."

The final image must be flawlessly executed in LANDSCAPE orientation, evoking an immediate 'must-have' response, making it irresistibly clickable and truly stunning for the viewer."
"""
                
                # Open all downloaded images for the API
                image_handles = []
                try:
                    for img_path in image_files:
                        image_handles.append(open(img_path, "rb"))
                    
                    # Generate the combo image using GPT-Image-1
                    logger.info(f"Generating landscape combo image using GPT-Image-1...")
                    result = self.client.images.edit(
                        model="gpt-image-1",
                        image=image_handles,
                        prompt=enhanced_prompt,
                        size="1536x1024"  # Landscape dimensions for better email display
                    )
                    
                    image_base64 = result.data[0].b64_json
                    image_bytes = base64.b64decode(image_base64)
                    
                    # Create output directory if it doesn't exist
                    os.makedirs('combo_images', exist_ok=True)
                    
                    # Save the generated image with random filename
                    import uuid
                    random_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
                    filename = f"combo_{random_id}.png"
                    output_path = os.path.join('combo_images', filename)
                    
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)
                    
                    logger.info(f"Successfully generated combo image: {output_path}")
                    
                    # Upload to S3
                    s3_key = f"combo_images/{filename}"
                    s3_url = self.upload_to_s3(output_path, s3_key)
                    
                    return {
                        'combo_id': combo_id,
                        'combo_name': combo_name,
                        'generated_image_path': output_path,
                        's3_url': s3_url,
                        's3_key': s3_key,
                        'products_used': [p.get('product_id', '') for p in products],
                        'product_descriptions': product_descriptions,
                        'image_prompt': enhanced_prompt,
                        'generation_success': True
                    }
                    
                finally:
                    # Close all image handles
                    for handle in image_handles:
                        handle.close()
            
        except Exception as e:
            logger.error(f"Failed to generate combo image for {combo.get('combo_name', 'Unknown')}: {str(e)}")
            return {
                'combo_id': combo.get('combo_id', 'unknown'),
                'combo_name': combo.get('combo_name', 'Unknown Combo'),
                'generated_image_path': '',
                's3_url': '',
                's3_key': '',
                'products_used': [p.get('product_id', '') for p in combo.get('products', [])],
                'product_descriptions': '',
                'image_prompt': '',
                'generation_success': False,
                'error': str(e)
            }
    
    def process_combo_suggestions(self, combo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process combo suggestions and generate images for each combo.
        
        Args:
            combo_data: Dictionary containing combo suggestions
            
        Returns:
            Dictionary with generated images for each combo
        """
        try:
            logger.info("Processing combo suggestions for image generation...")
            
            results = {
                'total_users': combo_data.get('total_users', 0),
                'total_combos': 0,
                'generated_images': [],
                'errors': []
            }
            
            for user_combo in combo_data.get('user_combos', []):
                user_id = user_combo.get('user_id', 'unknown')
                combinations = user_combo.get('combinations', [])
                
                logger.info(f"Generating images for user {user_id} with {len(combinations)} combinations")
                
                for combo in combinations:
                    try:
                        image_result = self.generate_combo_image(combo)
                        results['generated_images'].append(image_result)
                        results['total_combos'] += 1
                        
                    except Exception as e:
                        error_msg = f"Failed to generate image for combo {combo.get('combo_id', 'unknown')}: {str(e)}"
                        logger.error(error_msg)
                        results['errors'].append(error_msg)
            
            logger.info(f"Generated images for {results['total_combos']} combos")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process combo suggestions: {str(e)}")
            return {
                'total_users': 0,
                'total_combos': 0,
                'generated_images': [],
                'errors': [str(e)]
            }

if __name__ == "__main__":
    # Test the combo image creator
    creator = ComboImageCreator()
    
    # Load combo suggestions from JSON file
    try:
        with open('combo_suggestions.json', 'r') as f:
            combo_data = json.load(f)
        
        # Process all combos
        results = creator.process_combo_suggestions(combo_data)
        
        print(f"Generated {results['total_combos']} combo images:")
        for img_result in results['generated_images']:
            if img_result['generation_success']:
                print(f"✅ {img_result['combo_name']}: {img_result['generated_image_path']}")
            else:
                print(f"❌ {img_result['combo_name']}: {img_result.get('error', 'Unknown error')}")
        
        if results['errors']:
            print(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
                
    except FileNotFoundError:
        print("combo_suggestions.json not found. Please ensure the file exists.")
    except Exception as e:
        print(f"Error processing combo suggestions: {str(e)}")