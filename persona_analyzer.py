#!/usr/bin/env python3
"""
Style Persona Analyzer
Analyzes images and returns style persona traits using OpenAI's GPT-4 Vision API.
"""

import os
import json
import csv
import base64
import requests
import tempfile
from typing import List, Dict, Any
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class PersonaAnalyzer:
    def __init__(self, api_key: str = None):
        """Initialize the PersonaAnalyzer with OpenAI client."""
        # Get API key from parameter, environment variable, or .env file
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass as parameter.")
        
        self.client = OpenAI(api_key=api_key)
        
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for OpenAI API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def download_image_from_url(self, image_url: str) -> str:
        """Download image from URL and save to temporary file."""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            raise Exception(f"Failed to download image from {image_url}: {str(e)}")
    
    def analyze_image_from_url(self, image_url: str) -> Dict[str, Any]:
        """
        Analyze an image from URL and return style persona traits.
        
        Args:
            image_url: URL of the image to analyze
            
        Returns:
            Dictionary containing the style persona analysis
        """
        temp_file_path = None
        try:
            # Download image to temporary file
            temp_file_path = self.download_image_from_url(image_url)
            
            # Analyze the downloaded image
            result = self.analyze_image(temp_file_path)
            
            # Add URL information to result
            result['image_url'] = image_url
            
            return result
            
        except Exception as e:
            return {
                "error": f"Failed to analyze image from URL {image_url}: {str(e)}",
                "image_url": image_url
            }
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single image and return style persona traits.
        
        Args:
            image_path: Path to the image file (supports AVIF and other formats)
            
        Returns:
            Dictionary containing the style persona analysis
        """
        try:
            # Encode the image
            base64_image = self.encode_image(image_path)

            system_prompt = """You are a fashion style classification assistant. 
                Your role is to analyze clothing and appearance in an image and strictly classify the outfit using predefined schema categories. 
                Always return JSON output that exactly matches the given schema — no explanations, no additional text.
            """

            
            # Create the prompt
            prompt = """
                Analyze the outfit in the provided image and classify it across the following categories:
                - audience_segment
                - product_category
                - style_category
                - fashion_persona
                - formality_level
                - silhouette_preference
                - sleeve_style
                - neckline_type
                - length_type
                - pattern_print
                - texture_fabric_choice
                - texture_energy
                - color_palette_mood
                - occasion_context
                - confidence_level
                - personality_projection
                
                Rules:
                1. For each category, return one value from the predefined enum list in the schema.
                2. If the trait cannot be confidently determined from visible style elements, return "None".
                3. Focus only on clothing, accessories, patterns, textures, and colors — ignore background or facial expressions.
                4. Output must be a valid JSON object conforming exactly to the schema.
            """
            
            # Make the API call
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "persona_identification",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "audience_segment": {
                                    "type": "string",
                                    "enum": ["Women", "Men", "Kids", "Unisex", "None"]
                                },
                                "product_category": {
                                    "type": "string",
                                    "enum": ["Tops", "Bottoms", "Dresses & Jumpsuits", "Ethnicwear", "Fusionwear", "Outerwear", "Footwear", "Accessories", "Activewear", "Loungewear", "Innerwear & Sleepwear", "Occasionwear", "Formalwear", "None"]
                                },
                                "style_category": {
                                    "type": "string",
                                    "enum": ["Athletic", "Casualwear", "Ethnic / Fusion", "Formalwear", "Luxury", "Smart Casual", "Streetwear","None"]
                                },
                                "fashion_persona": {
                                    "type": "string",
                                    "enum": ["Avant-garde", "Bohemian", "Classic", "Edgy", "Minimalist", "Sporty", "Trendy","None"]
                                },
                                "formality_level": {
                                    "type": "string",
                                    "enum": ["Formal", "Opulent", "Relaxed", "Semi-formal","None"]
                                },
                                "silhouette_preference": {
                                    "type": "string",
                                    "enum": ["Flowy", "Oversized", "Regular", "Relaxed", "Slim Fit", "Structured", "Tailored","None"]
                                },
                                "sleeve_style": {
                                    "type": "string",
                                    "enum": ["Cap Sleeve", "Extended / Batwing", "Full Sleeve", "Half Sleeve", "Raglan", "Sleeveless", "Three-Quarter","None"]
                                },
                                "neckline_type": {
                                    "type": "string",
                                    "enum": ["Collared", "Henley", "High Neck / Turtle", "Mandarin", "Off-Shoulder", "Polo", "Round Neck", "Scoop", "V-Neck","None"]
                                },
                                "length_type": {
                                    "type": "string",
                                    "enum": ["Ankle Length", "Crop", "Hip Length", "Knee Length", "Maxi / Floor Length", "Thigh Length", "Waist Length","None"]
                                },
                                "pattern_print": {
                                    "type": "string",
                                    "enum": ["Abstract", "Checked", "Colorblock", "Embroidered", "Floral", "Geometric", "Graphic", "Mixed Motif", "Solid", "Striped", "Textured", "Tie-Dye","None"]
                                },
                                "texture_fabric_choice": {
                                    "type": "string",
                                    "enum": ["Blended", "Cotton", "Denim", "Leather", "Linen", "Polyester", "Rayon", "Sheer / Mesh", "Silk / Satin", "Technical Synthetic", "Wool / Knit","None"]
                                },
                                "texture_energy": {
                                    "type": "string",
                                    "enum": ["Crisp", "Fluid", "Glossy", "Matte", "Rough / Textured", "Soft","None"]
                                },
                                "color_palette_mood": {
                                    "type": "string",
                                    "enum": ["Bright", "Contrasting", "Dark", "Earthy", "Light", "Monochrome", "Neutral", "Pastel","None"]
                                },
                                "occasion_context": {
                                    "type": "string",
                                    "enum": ["Ceremonial / Wedding", "Everyday Casual", "Festive / Cultural", "Lounge / Leisure", "Party / Evening", "Sports / Performance", "Travel Ready", "Work / Professional","None"]
                                },
                                "confidence_level": {
                                    "type": "string",
                                    "enum": ["Bold", "Commanding", "Effortless", "Playful", "Reserved","None"]
                                },
                                "personality_projection": {
                                    "type": "string",
                                    "enum": ["Dreamer", "Elegant Minimalist", "Free Spirit", "Leader", "Rebel", "Romantic", "Trendsetter", "Visionary","None"]
                                },
                            },
                            "required": [
                                "audience_segment", "product_category", "style_category", "fashion_persona", "formality_level", "silhouette_preference",
                                "sleeve_style", "neckline_type", "length_type", "pattern_print",
                                "texture_fabric_choice", "texture_energy", "color_palette_mood", "occasion_context",
                                "confidence_level", "personality_projection",
                            ],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "error": f"Failed to analyze image {image_path}: {str(e)}",
                "image_path": image_path
            }
    
    def analyze_images_folder(self, images_folder: str) -> List[Dict[str, Any]]:
        """
        Analyze all images in a folder and return results.
        
        Args:
            images_folder: Path to the folder containing images
            
        Returns:
            List of dictionaries containing analysis results for each image
        """
        results = []
        images_folder = Path(images_folder)
        
        if not images_folder.exists():
            raise FileNotFoundError(f"Images folder not found: {images_folder}")
        
        # Supported image formats
        supported_formats = {'.avif', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in supported_formats:
            image_files.extend(images_folder.glob(f"*{ext}"))
            image_files.extend(images_folder.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No supported image files found in {images_folder}")
            return results
        
        print(f"Found {len(image_files)} image(s) to analyze...")
        
        for image_file in image_files:
            print(f"Analyzing: {image_file.name}")
            result = self.analyze_image(str(image_file))
            result["filename"] = image_file.name
            # Extract product_id from filename (remove extension)
            product_id = image_file.stem
            result["product_id"] = product_id
            results.append(result)
        
        return results
    
    def _get_file_extension(self, product_id: str) -> str:
        """Determine the file extension for a given product ID"""
        possible_extensions = ['.png', '.jpg', '.webp']
        images_dir = "images"  # Default images directory
        
        for ext in possible_extensions:
            filename = f"{product_id}{ext}"
            if os.path.exists(os.path.join(images_dir, filename)):
                return ext
        
        # Default to .png if not found
        return '.png'
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "persona_analysis_results.json"):
        """Save analysis results to JSON, Elasticsearch dev tools, and CSV files."""
        
        # Convert results to the new format
        formatted_results = []
        for item in results:
            product_id = item['product_id']
            
            # Determine file extension based on available images
            file_ext = self._get_file_extension(product_id)
            s3_url = f"https://hacka-image-bucket.s3.us-east-1.amazonaws.com/images/{product_id}{file_ext}"
            
            # Create metadata object (excluding product_id and filename)
            metadata = {k: v for k, v in item.items() if k not in ['product_id', 'filename']}
            
            # Create the formatted product object
            product = {
                "product_id": product_id,
                "s3_url": s3_url,
                "metadata": metadata
            }
            formatted_results.append(product)
        
        # Save JSON file (simple array format)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to {output_file}")
        
        # Save Elasticsearch dev tools format
        es_file = output_file.replace('.json', '_elasticsearch.txt')
        with open(es_file, 'w', encoding='utf-8') as f:
            for product in formatted_results:
                f.write("POST products/_doc\n")
                f.write("{\n")
                f.write(f'  "product_id": "{product["product_id"]}",\n')
                f.write(f'  "s3_url": "{product["s3_url"]}",\n')
                f.write('  "metadata": {\n')
                
                # Write metadata fields
                metadata_items = list(product["metadata"].items())
                for i, (key, value) in enumerate(metadata_items):
                    # Escape quotes in values
                    escaped_value = str(value).replace('"', '\\"')
                    f.write(f'    "{key}": "{escaped_value}"')
                    if i < len(metadata_items) - 1:
                        f.write(',')
                    f.write('\n')
                
                f.write('  }\n')
                f.write('}\n\n')
        print(f"Elasticsearch dev tools format saved to {es_file}")
        
        # Save CSV file
        csv_file = output_file.replace('.json', '.csv')
        self.save_results_csv(results, csv_file)
    
    def save_results_csv(self, results: List[Dict[str, Any]], csv_file: str = "persona_analysis_results.csv"):
        """Save analysis results to a CSV file."""
        if not results:
            print("No results to save to CSV")
            return
        
        # Define the fieldnames based on our schema
        fieldnames = [
            'filename', 'audience_segment', 'product_category', 'style_category', 'fashion_persona', 'formality_level',
            'silhouette_preference', 'sleeve_style', 'neckline_type', 'length_type',
            'pattern_print', 'texture_fabric_choice', 'texture_energy', 'color_palette_mood',
            'occasion_context', 'confidence_level', 'personality_projection'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Create a clean row with only the fields we want
                row = {}
                for field in fieldnames:
                    if field in result:
                        row[field] = result[field]
                    else:
                        row[field] = 'N/A'
                
                # Handle error cases
                if 'error' in result:
                    row['filename'] = result.get('filename', 'Unknown')
                    for field in fieldnames[1:-1]:  # All fields except filename
                        row[field] = 'ERROR'
                
                writer.writerow(row)
        
        print(f"CSV results saved to {csv_file}")


def main():
    """Main function to run the persona analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze style persona from images")
    parser.add_argument("--images-folder", "-i", default="images", 
                       help="Path to folder containing images (default: images)")
    parser.add_argument("--output", "-o", default="persona_analysis_results.json",
                       help="Output JSON file path (default: persona_analysis_results.json)")
    parser.add_argument("--api-key", "-k", 
                       help="OpenAI API key (or set OPENAI_API_KEY in .env file)")
    
    args = parser.parse_args()
    
    # Get API key from argument, environment variable, or .env file
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required.")
        print("Please set OPENAI_API_KEY in a .env file or use --api-key")
        print("\nTo create .env file:")
        print("echo 'OPENAI_API_KEY=your-api-key-here' > .env")
        return
    
    # Check if images folder exists
    if not os.path.exists(args.images_folder):
        print(f"Error: Images folder '{args.images_folder}' not found.")
        print("Please create the folder and add some images, or specify a different folder with --images-folder")
        return
    
    try:
        # Initialize analyzer
        print("Initializing Persona Analyzer...")
        analyzer = PersonaAnalyzer(api_key)
        
        # Analyze images
        print(f"Analyzing images in '{args.images_folder}' folder...")
        results = analyzer.analyze_images_folder(args.images_folder)
        
        if results:
            # Save results
            analyzer.save_results(results, args.output)
            
            # Print summary
            print(f"\n✅ Analysis complete! Processed {len(results)} image(s).")
            print(f"📄 JSON results saved to: {args.output}")
            print(f"📊 CSV results saved to: {args.output.replace('.json', '.csv')}")
            
            # Print results summary
            print(f"\n📊 Analysis Summary:")
            print("=" * 50)
            
            for i, result in enumerate(results, 1):
                if "error" not in result:
                    print(f"\n{i}. {result['filename']}:")
                    print(f"   👥 Audience Segment: {result.get('audience_segment', 'N/A')}")
                    print(f"   🏷️  Product Category: {result.get('product_category', 'N/A')}")
                    print(f"   🏷️  Style Category: {result.get('style_category', 'N/A')}")
                    print(f"   👗 Fashion Persona: {result.get('fashion_persona', 'N/A')}")
                    print(f"   🎭 Formality Level: {result.get('formality_level', 'N/A')}")
                    print(f"   ✂️  Silhouette: {result.get('silhouette_preference', 'N/A')}")
                    print(f"   👕 Sleeve Style: {result.get('sleeve_style', 'N/A')}")
                    print(f"   👖 Neckline Type: {result.get('neckline_type', 'N/A')}")
                    print(f"   👖 Length Type: {result.get('length_type', 'N/A')}")
                    print(f"   👖 Pattern Print: {result.get('pattern_print', 'N/A')}")
                    print(f"   👖 Texture Fabric Choice: {result.get('texture_fabric_choice', 'N/A')}")
                    print(f"   👖 Texture Energy: {result.get('texture_energy', 'N/A')}")
                    print(f"   👖 Confidence Level: {result.get('confidence_level', 'N/A')}")
                    print(f"   🎨 Color Palette: {result.get('color_palette_mood', 'N/A')}")
                    print(f"   🎪 Occasion: {result.get('occasion_context', 'N/A')}")
                    print(f"   💫 Personality: {result.get('personality_projection', 'N/A')}")
                    print(f"   👥 Audience Segment: {result.get('audience_segment', 'N/A')}")
                    print(f"   🏷️  Product Category: {result.get('product_category', 'N/A')}")
                    
                else:
                    print(f"\n{i}. ❌ Error with {result.get('filename', 'Unknown')}: {result['error']}")
            
            print(f"\n🎉 All done! Check both files for results:")
            print(f"   📄 JSON: {args.output}")
            print(f"   📊 CSV: {args.output.replace('.json', '.csv')}")
        else:
            print("❌ No images were processed.")
            print("Please add some images (AVIF, JPG, PNG, etc.) to the images folder.")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure your OpenAI API key is correct")
        print("2. Check that you have images in the images folder")
        print("3. Ensure you have an internet connection")


if __name__ == "__main__":
    main()
