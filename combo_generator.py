#!/usr/bin/env python3
"""
Product Combo Generator
Uses LLM to analyze similar products and suggest good combinations for customers.
"""

import json
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComboGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the ComboGenerator with OpenAI client."""
        # Get API key from parameter, environment variable, or .env file
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass as parameter.")
        
        self.client = OpenAI(api_key=api_key)
    
    def extract_products_for_combo(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract only similar products (excluding original products) for combo generation.
        
        Args:
            analysis_results: Results from the customer analysis Lambda function
            
        Returns:
            List of similar products with their metadata
        """
        similar_products = []
        
        for user_result in analysis_results.get('results', []):
            user_id = user_result.get('user_id')
            
            for product in user_result.get('processed_products', []):
                # Only add similar products (exclude original products)
                for similar in product.get('similar_products', []):
                    similar_products.append({
                        'product_id': similar.get('product_id'),
                        's3_url': similar.get('s3_url'),
                        'user_id': user_id,
                        'metadata': similar.get('metadata', {}),
                        'similarity_score': similar.get('similarity_score', 0.0),
                        'product_type': 'similar'
                    })
        
        logger.info(f"Extracted {len(similar_products)} similar products for combo generation (excluding original products)")
        return similar_products
    
    def create_product_summary(self, products: List[Dict[str, Any]]) -> str:
        """
        Create a text summary of products for LLM analysis.
        
        Args:
            products: List of products with metadata
            
        Returns:
            Formatted text summary
        """
        summary_parts = []
        
        for i, product in enumerate(products, 1):
            metadata = product.get('metadata', {})
            product_type = product.get('product_type', 'unknown')
            similarity_score = product.get('similarity_score', 0.0)
            
            summary_parts.append(f"\n{i}. Product ID: {product['product_id']}")
            summary_parts.append(f"   Type: {product_type} (similarity: {similarity_score:.3f})")
            # summary_parts.append(f"   S3 URL: {product['s3_url']}")
            summary_parts.append(f"   Audience: {metadata.get('audience_segment', 'N/A')}")
            summary_parts.append(f"   Category: {metadata.get('product_category', 'N/A')}")
            summary_parts.append(f"   Style: {metadata.get('style_category', 'N/A')}")
            summary_parts.append(f"   Persona: {metadata.get('fashion_persona', 'N/A')}")
            summary_parts.append(f"   Formality: {metadata.get('formality_level', 'N/A')}")
            summary_parts.append(f"   Silhouette: {metadata.get('silhouette_preference', 'N/A')}")
            summary_parts.append(f"   Sleeve: {metadata.get('sleeve_style', 'N/A')}")
            summary_parts.append(f"   Neckline: {metadata.get('neckline_type', 'N/A')}")
            summary_parts.append(f"   Length: {metadata.get('length_type', 'N/A')}")
            summary_parts.append(f"   Colors: {metadata.get('color_palette_mood', 'N/A')}")
            summary_parts.append(f"   Occasion: {metadata.get('occasion_context', 'N/A')}")
            summary_parts.append(f"   Pattern: {metadata.get('pattern_print', 'N/A')}")
            summary_parts.append(f"   Fabric: {metadata.get('texture_fabric_choice', 'N/A')}")
        
        return "\n".join(summary_parts)
    
    def generate_combos(self, 
                       products: List[Dict[str, Any]], 
                       max_combos: int = 5,
                       combo_size: int = 3) -> List[Dict[str, Any]]:
        """
        Generate product combinations using LLM.
        
        Args:
            products: List of products to combine
            max_combos: Maximum number of combos to generate
            combo_size: Number of products per combo
            
        Returns:
            List of suggested combinations
        """
        try:
            # Create product summary
            product_summary = self.create_product_summary(products)
            
            system_prompt = """You are a fashion styling expert. Your role is to analyze fashion products and suggest stylish combinations that work well together.

Rules:
1. Consider color harmony, style compatibility, and occasion appropriateness
2. Mix different product categories (tops, bottoms, accessories) when possible
3. Ensure the combination tells a cohesive style story
4. Consider the target audience and formality level
5. Avoid clashing patterns or colors
6. Return valid JSON format as specified in the schema
7. Focus on creating visually appealing and practical combinations"""

            prompt = f"""
Analyze the following fashion products and suggest {max_combos} stylish combinations of {combo_size} products each.

Products to analyze:
{product_summary}

For each combination, consider:
- Color harmony and contrast
- Style consistency (casual, formal, etc.)
- Product category variety (tops, bottoms, accessories)
- Occasion appropriateness
- Target audience compatibility
- Pattern and texture coordination

Generate combinations that would look great together in a styled photo.
"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fashion_combinations",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "combinations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "combo_id": {
                                                "type": "string",
                                                "description": "Unique identifier for this combination"
                                            },
                                            "combo_name": {
                                                "type": "string",
                                                "description": "Creative name for this combination"
                                            },
                                            "products": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "product_id": {
                                                            "type": "string",
                                                            "description": "Product ID from the input"
                                                        },
                                                        "role": {
                                                            "type": "string",
                                                            "description": "Role in the outfit (e.g., 'top', 'bottom', 'accessory', 'outerwear')"
                                                        },
                                                        "styling_notes": {
                                                            "type": "string",
                                                            "description": "How to style this item in the combination"
                                                        }
                                                    },
                                                    "required": ["product_id", "role", "styling_notes"],
                                                    "additionalProperties": False
                                                },
                                                "minItems": combo_size,
                                                "maxItems": combo_size
                                            },
                                            "style_description": {
                                                "type": "string",
                                                "description": "Description of the overall style and vibe"
                                            },
                                            "occasion": {
                                                "type": "string",
                                                "description": "Best occasion for this combination"
                                            },
                                            "color_palette": {
                                                "type": "string",
                                                "description": "Main colors in this combination"
                                            },
                                            "confidence_score": {
                                                "type": "number",
                                                "description": "Confidence in this combination (0.0 to 1.0)"
                                            }
                                        },
                                        "required": ["combo_id", "combo_name", "products", "style_description", "occasion", "color_palette", "confidence_score"],
                                        "additionalProperties": False
                                    },
                                    "minItems": 1,
                                    "maxItems": max_combos
                                }
                            },
                            "required": ["combinations"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                temperature=0.7
            )
            
            result = json.loads(response.choices[0].message.content)
            combinations = result['combinations']
            
            # Add S3 URLs to each product in the combinations
            for combo in combinations:
                for product in combo.get('products', []):
                    product_id = product.get('product_id')
                    # Find the original product data to get S3 URL
                    for original_product in products:
                        if original_product.get('product_id') == product_id:
                            product['s3_url'] = original_product.get('s3_url', '')
                            break
            
            logger.info(f"Generated {len(combinations)} combinations")
            return combinations
            
        except Exception as e:
            logger.error(f"Failed to generate combinations: {str(e)}")
            return []
    
    def process_customer_analysis(self, 
                                analysis_results: Dict[str, Any],
                                max_combos: int = 5,
                                combo_size: int = 3) -> Dict[str, Any]:
        """
        Process customer analysis results and generate product combinations.
        
        Args:
            analysis_results: Results from lambda_customer_analysis
            max_combos: Maximum number of combos to generate per user
            combo_size: Number of products per combo
            
        Returns:
            Dictionary with combo suggestions for each user
        """
        try:
            logger.info("Processing customer analysis for combo generation...")
            
            results = {
                'total_users': analysis_results.get('total_users', 0),
                'total_products': analysis_results.get('total_products_processed', 0),
                'user_combos': []
            }
            
            # Process each user's products
            for user_result in analysis_results.get('results', []):
                user_id = user_result.get('user_id')
                logger.info(f"Generating combos for user {user_id}")
                
                # Extract only similar products for this user (exclude original products)
                user_products = []
                for product in user_result.get('processed_products', []):
                    # Only add similar products (exclude original products)
                    for similar in product.get('similar_products', []):
                        user_products.append({
                            'product_id': similar.get('product_id'),
                            's3_url': similar.get('s3_url'),
                            'user_id': user_id,
                            'metadata': similar.get('metadata', {}),
                            'similarity_score': similar.get('similarity_score', 0.0),
                            'product_type': 'similar'
                        })
                
                # Generate combinations for this user
                if user_products:
                    combinations = self.generate_combos(user_products, max_combos, combo_size)
                    
                    user_combo_result = {
                        'user_id': user_id,
                        'total_products_available': len(user_products),
                        'combinations': combinations,
                        'combo_summary': {
                            'total_combos': len(combinations),
                            'avg_confidence': sum(c.get('confidence_score', 0) for c in combinations) / len(combinations) if combinations else 0
                        }
                    }
                    
                    results['user_combos'].append(user_combo_result)
                    logger.info(f"Generated {len(combinations)} combinations for user {user_id}")
                else:
                    logger.warning(f"No products available for combo generation for user {user_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to process customer analysis: {str(e)}")
            return {
                'error': str(e),
                'total_users': 0,
                'total_products': 0,
                'user_combos': []
            }

def main():
    """Main function for testing combo generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate fashion product combinations")
    parser.add_argument("--input-file", "-i", 
                       help="JSON file with customer analysis results")
    parser.add_argument("--max-combos", "-c", type=int, default=5,
                       help="Maximum number of combinations to generate")
    parser.add_argument("--combo-size", "-s", type=int, default=3,
                       help="Number of products per combination")
    parser.add_argument("--output", "-o", default="combo_suggestions.json",
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    try:
        # Initialize combo generator
        generator = ComboGenerator()
        
        # Load analysis results
        if args.input_file:
            with open(args.input_file, 'r') as f:
                analysis_results = json.load(f)
        else:
            # Use test data
            print("Using test data...")
            analysis_results = {
                "total_users": 1,
                "total_products_processed": 4,
                "results": [
                    {
                        "user_id": "46382",
                        "total_products": 4,
                        "processed_products": [
                            {
                                "original_product": {
                                    "product_id": "8909230826683",
                                    "s3_url": "https://hacka-image-bucket.s3.us-east-1.amazonaws.com/customer_products/8909230826683.png",
                                    "user_id": "46382"
                                },
                                "persona_analysis": {
                                    "product_id": "8909230826683",
                                    "s3_url": "https://hacka-image-bucket.s3.us-east-1.amazonaws.com/customer_products/8909230826683.png",
                                    "metadata": {
                                        "audience_segment": "Men",
                                        "product_category": "Tops",
                                        "style_category": "Casualwear",
                                        "fashion_persona": "Minimalist",
                                        "formality_level": "Relaxed",
                                        "color_palette_mood": "Neutral"
                                    },
                                    "analysis_success": True
                                },
                                "similar_products": [
                                    {
                                        "product_id": "similar_1",
                                        "s3_url": "https://example.com/similar1.jpg",
                                        "metadata": {
                                            "audience_segment": "Men",
                                            "product_category": "Bottoms",
                                            "style_category": "Casualwear",
                                            "fashion_persona": "Minimalist",
                                            "formality_level": "Relaxed",
                                            "color_palette_mood": "Neutral"
                                        },
                                        "similarity_score": 0.85
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        
        # Generate combinations
        combo_results = generator.process_customer_analysis(
            analysis_results, 
            max_combos=args.max_combos,
            combo_size=args.combo_size
        )
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(combo_results, f, indent=2)
        
        print(f"✅ Generated combinations saved to {args.output}")
        print(f"📊 Summary:")
        print(f"   - Users processed: {combo_results.get('total_users', 0)}")
        print(f"   - Total products: {combo_results.get('total_products', 0)}")
        print(f"   - User combos: {len(combo_results.get('user_combos', []))}")
        
        for user_combo in combo_results.get('user_combos', []):
            print(f"\n👤 User {user_combo['user_id']}:")
            print(f"   - Products available: {user_combo['total_products_available']}")
            print(f"   - Combinations generated: {user_combo['combo_summary']['total_combos']}")
            print(f"   - Average confidence: {user_combo['combo_summary']['avg_confidence']:.2f}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
