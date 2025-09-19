#!/usr/bin/env python3
"""
Fashion Search API
FastAPI application that analyzes images for fashion persona and finds similar products.
"""

import os
import json
import base64
import tempfile
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing modules
from persona_analyzer import PersonaAnalyzer
from simple_vector_search import SimpleVectorSearch
from knn_vector_search import KNNVectorSearch
from combo_generator import ComboGenerator
from combo_image_creator import ComboImageCreator
from lambda_customer_analysis import lambda_handler
import json
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for initialized components
persona_analyzer = None
simple_vector_search = None
knn_vector_search = None
combo_generator = None
combo_image_creator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup components."""
    global persona_analyzer, simple_vector_search, knn_vector_search, combo_generator, combo_image_creator
    
    # Startup - Initialize components lazily
    logger.info("🚀 Fashion Search API starting...")
    
    # Initialize components as None - they'll be created when needed
    persona_analyzer = None
    simple_vector_search = None
    knn_vector_search = None
    combo_generator = None
    combo_image_creator = None
    
    logger.info("✅ Fashion Search API ready!")
    
    yield
    
    # Shutdown (if needed)
    logger.info("Shutting down Fashion Search API...")

def get_persona_analyzer():
    """Get or create persona analyzer."""
    global persona_analyzer
    if persona_analyzer is None:
        persona_analyzer = PersonaAnalyzer()
    return persona_analyzer

def get_combo_generator():
    """Get or create combo generator."""
    global combo_generator
    if combo_generator is None:
        combo_generator = ComboGenerator()
    return combo_generator

def get_combo_image_creator():
    """Get or create combo image creator."""
    global combo_image_creator
    if combo_image_creator is None:
        combo_image_creator = ComboImageCreator()
    return combo_image_creator

def get_vector_search(search_method: str):
    """Get or create vector search instance."""
    global simple_vector_search, knn_vector_search
    
    opensearch_endpoint = os.getenv("OPENSEARCH_ENDPOINT")
    if not opensearch_endpoint:
        raise ValueError("OPENSEARCH_ENDPOINT environment variable is required")
    
    opensearch_index = os.getenv("OPENSEARCH_INDEX", "products")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    if search_method.lower() == "knn":
        if knn_vector_search is None:
            knn_vector_search = KNNVectorSearch(
                opensearch_endpoint=opensearch_endpoint,
                opensearch_index=opensearch_index,
                aws_region=aws_region
            )
        return knn_vector_search
    elif search_method.lower() == "simple":
        if simple_vector_search is None:
            simple_vector_search = SimpleVectorSearch(
                opensearch_endpoint=opensearch_endpoint,
                opensearch_index=opensearch_index,
                aws_region=aws_region
            )
        return simple_vector_search
    else:
        raise ValueError(f"Invalid search method: {search_method}. Use 'knn' or 'simple'")

# Initialize FastAPI app
app = FastAPI(
    title="Fashion Search API",
    description="Analyze fashion images and find similar products using vector similarity search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ProductSearchRequest(BaseModel):
    image_base64: str
    max_results: int = 5
    search_method: str = "knn"  # "knn" or "simple"

class ProductSearchResponse(BaseModel):
    analyzed_product: Dict[str, Any]
    similar_products: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]

class TextSearchRequest(BaseModel):
    query_text: str
    max_results: int = 5
    search_method: str = "knn"  # "knn" or "simple"

class ProductDataSearchRequest(BaseModel):
    product_data: Dict[str, Any]
    max_results: int = 5
    search_method: str = "knn"  # "knn" or "simple"

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]

class UserEmailRequest(BaseModel):
    user_id: str
    max_combos: int = 2
    combo_size: int = 3
    max_similar_products: int = 2

class UserEmailResponse(BaseModel):
    email_html: str

class TestEmailRequest(BaseModel):
    use_existing_combos: bool = True



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        components={
            "api": "✅ Ready",
            "status": "✅ Running"
        }
    )

# This function is now defined above, removing the duplicate

@app.post("/search/image", response_model=ProductSearchResponse)
async def search_by_image(
    image: UploadFile = File(..., description="Image file to analyze"),
    max_results: int = Form(5, description="Maximum number of results to return"),
    search_method: str = Form("knn", description="Search method: 'knn' or 'simple'")
):
    """
    Analyze an image for fashion persona and find similar products.
    
    Args:
        image: Image file to analyze
        max_results: Maximum number of similar products to return
        search_method: Search method to use ('knn' or 'simple')
        
    Returns:
        Analyzed product data and similar products
    """
    try:
        persona_analyzer = get_persona_analyzer()
        vector_search = get_vector_search(search_method)
        
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{image.filename.split('.')[-1]}") as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Analyze the image for persona
            logger.info(f"Analyzing image: {image.filename}")
            analysis_result = persona_analyzer.analyze_image(temp_file_path)
            
            if not analysis_result:
                raise HTTPException(status_code=400, detail="Failed to analyze image")
            
            # Extract metadata from analysis result
            # The analyze_image method returns metadata directly, not wrapped in 'metadata' key
            metadata = analysis_result
            product_id = analysis_result.get('product_id', 'unknown')
            
            # Create product data structure for vector search
            product_data = {
                "product_id": product_id,
                "metadata": metadata
            }
            
            # Search for similar products
            logger.info(f"Searching for similar products to: {product_id} using {search_method} method")
            if search_method.lower() == "knn":
                similar_products = vector_search.search_by_product_data_knn(
                    product_data, 
                    max_results
                )
            else:  # simple
                similar_products = vector_search.search_by_product_data(
                    product_data, 
                    max_results
                )
            
            # Prepare response
            response = ProductSearchResponse(
                analyzed_product={
                    "product_id": product_id,
                    "metadata": metadata,
                    "filename": analysis_result.get('filename', image.filename)
                },
                similar_products=similar_products,
                search_metadata={
                    "total_results": len(similar_products),
                    "search_method": f"{search_method}_vector_similarity",
                    "model_used": "text-embedding-3-small"
                }
            )
            
            logger.info(f"✅ Found {len(similar_products)} similar products")
            return response
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/generate-user-email", response_model=UserEmailResponse)
async def generate_user_email(request: UserEmailRequest):
    """
    Generate personalized email with combo images for a specific user.
    
    This endpoint performs the complete flow:
    1. Get products for the user from CSV
    2. Analyze personas/metadata using LLM
    3. Convert metadata to embeddings
    4. Search for similar products in OpenSearch
    5. Generate combos using LLM
    6. Generate combo images
    7. Create HTML email template
    
    Args:
        request: User email request with user_id and parameters
        
    Returns:
        User email response with combo images and HTML template
    """
    try:
        # Initialize components lazily
        persona_analyzer = get_persona_analyzer()
        knn_vector_search = get_vector_search("knn")
        combo_generator = get_combo_generator()
        combo_image_creator = get_combo_image_creator()
        
        user_id = request.user_id
        logger.info(f"Starting email generation for user: {user_id}")
        
        # Step 1: Get products for the user from CSV
        logger.info(f"Step 1: Getting products for user {user_id}")
        user_products = []
        
        # Read CSV file to get user products
        csv_file_path = "customer_data.csv"
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=404, detail="Customer data CSV not found")
        
        with open(csv_file_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                if row['user_id'] == user_id:
                    user_products.append({
                        'user_id': row['user_id'],
                        'product_id': row['product_id'],
                        's3_url': row['s3_url']
                    })
        
        if not user_products:
            raise HTTPException(status_code=404, detail=f"No products found for user {user_id}")
        
        logger.info(f"Found {len(user_products)} products for user {user_id}")
        
        # Step 2-4: Process products (persona analysis + similar products)
        logger.info("Step 2-4: Processing products and finding similar items")
        
        # Create CSV content for lambda handler
        csv_content = "user_id,product_id,s3_url\n"
        for product in user_products:
            csv_content += f"{product['user_id']},{product['product_id']},{product['s3_url']}\n"
        
        # Process using lambda handler
        event = {
            'csv_data': csv_content,
            'max_similar_products': request.max_similar_products
        }
        
        lambda_response = lambda_handler(event, None)
        if lambda_response['statusCode'] != 200:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {lambda_response}")
        
        analysis_results = json.loads(lambda_response['body'])
        
        # Step 5: Generate combos
        logger.info("Step 5: Generating product combinations")
        combo_data = combo_generator.process_customer_analysis(
            analysis_results,
            max_combos=request.max_combos,
            combo_size=request.combo_size
        )
        
        # Step 6: Generate combo images
        logger.info("Step 6: Generating combo images")
        image_results = combo_image_creator.process_combo_suggestions(combo_data)
        
        # Collect combo image S3 URLs
        combo_images = []
        for img_result in image_results.get('generated_images', []):
            if img_result.get('generation_success', False):
                s3_url = img_result.get('s3_url', '')
                if s3_url:
                    combo_images.append(s3_url)
        
        # Step 7: Generate HTML email template
        logger.info("Step 7: Generating HTML email template")
        email_html = await generate_email_html(combo_images)
        
        # Save HTML to index.html
        try:
            with open('index.html', 'w', encoding='utf-8') as f:
                f.write(email_html)
            logger.info("📧 HTML template saved to index.html")
        except Exception as e:
            logger.warning(f"Failed to save HTML to index.html: {str(e)}")
        
        # Prepare response - return only HTML
        response = UserEmailResponse(
            email_html=email_html
        )
        
        logger.info(f"✅ Email generation completed for user {user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in email generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Email generation failed: {str(e)}")

async def generate_email_html(combo_images: List[str]) -> str:
    """
    Generate HTML email template using LLM based on combo images.
    
    Args:
        combo_images: List of paths to combo images
        
    Returns:
        HTML email template string
    """
    try:
        # Create prompt for email generation
        prompt = """You are a professional email marketing designer and fashion copywriter.

Task:
- Generate a clean, aesthetic HTML emailer template for a fashion campaign.
- I will provide one or more image URLs of outfits or fashion items.
- Use a fixed email structure (header, images, text, CTA, footer).
- Analyze the images to determine the style, vibe, and colors.
- Based on that, generate:
  1. A bold, stylish heading (2–4 words, brand-like).
  2. A short description (2–3 sentences) that matches the vibe of the items shown 
     (streetwear, casual, sporty, formal, etc.).
- Insert all provided images inside the email with proper spacing and text between them.
- Include a strong CTA button ("Shop the Look") below the text.
- Add a minimal footer with brand info.

Requirements:
- Output ONLY valid HTML code.
- Use inline styles (no external CSS).
- Keep the design premium, minimal, and mobile-friendly.
- Use plenty of white space for an aesthetic look.
- CTA button should have a bright accent color.
- If multiple image URLs are provided, display each image with:
  * Generous spacing between images (at least 40px margin)
  * A descriptive caption or text between each image
  * Each image should have its own section with proper spacing
- Structure should be: Header → Image 1 → Text/Caption → Image 2 → Text/Caption → Main Description → CTA → Footer
- YOU ONLY GENERATE HTML AND CSS, NO OTHER TEXT.

Input format:
COMBO IMAGES

output: HTML Template

Now generate the HTML email using the fixed template and dynamically create the heading and short description based on the images.

COMBO IMAGES:
"""
        
        # Add image S3 URLs to prompt
        for i, img_url in enumerate(combo_images, 1):
            prompt += f"Image {i}: {img_url}\n"
        
        # Try Claude first, fallback to OpenAI if authentication fails
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2000,
                temperature=0.7,
                system="You are a professional email marketing designer. Generate clean, aesthetic HTML email templates for fashion campaigns.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            html_content = response.content[0].text.strip()
            
        except Exception as claude_error:
            logger.warning(f"Claude API failed: {str(claude_error)}, falling back to OpenAI")
            
            # Fallback to OpenAI
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional email marketing designer. Generate clean, aesthetic HTML email templates for fashion campaigns."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            html_content = response.choices[0].message.content.strip()
        
        # Clean up the response to ensure it's valid HTML
        if html_content.startswith('```html'):
            html_content = html_content[7:]
        if html_content.endswith('```'):
            html_content = html_content[:-3]
        
        return html_content.strip()
        
    except Exception as e:
        logger.error(f"Error generating email HTML: {str(e)}")
        # Return a basic HTML template as fallback
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fashion Combo</title>
</head>
<body style="margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #f8f9fa;">
    <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="padding: 40px 30px; text-align: center;">
            <h1 style="color: #333; margin: 0 0 20px 0; font-size: 28px; font-weight: bold;">Style Combo</h1>
            <p style="color: #666; margin: 0 0 30px 0; font-size: 16px; line-height: 1.5;">Discover your perfect look with our curated fashion combinations.</p>
        </div>
        {"".join([f'''
        <div style="margin: 40px 0; padding: 0 20px;">
            <img src="{img_url}" style="width: 100%; max-width: 100%; height: auto; border-radius: 8px;" alt="Fashion Combo">
            <div style="text-align: center; margin: 20px 0 40px 0;">
                <h3 style="color: #333; margin: 0 0 10px 0; font-size: 18px; font-weight: 600;">Fashion Combo {i+1}</h3>
                <p style="color: #666; margin: 0; font-size: 14px; font-style: italic;">Curated style combination</p>
            </div>
        </div>
        ''' for i, img_url in enumerate(combo_images)])}
        <div style="padding: 40px 30px; text-align: center;">
            <a href="#" style="display: inline-block; background-color: #ff6b6b; color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: bold; margin: 30px 0;">Shop the Look</a>
        </div>
        <div style="background-color: #f8f9fa; padding: 20px; text-align: center; color: #666; font-size: 12px;">
            <p style="margin: 0;">© 2024 Fashion Brand. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""

@app.post("/test-email-generation", response_model=UserEmailResponse)
async def test_email_generation(request: TestEmailRequest):
    """
    Test endpoint that uses existing combo_suggestions.json to generate images and HTML.
    This is much faster for testing as it skips the full analysis flow.
    
    Args:
        request: Test email request
        
    Returns:
        HTML email template
    """
    try:
        combo_image_creator = get_combo_image_creator()
        
        logger.info("Starting test email generation using existing combo suggestions...")
        
        # Load existing combo suggestions
        combo_suggestions_file = "combo_suggestions.json"
        if not os.path.exists(combo_suggestions_file):
            raise HTTPException(status_code=404, detail="combo_suggestions.json not found. Please run the full flow first.")
        
        with open(combo_suggestions_file, 'r') as f:
            combo_data = json.load(f)
        
        logger.info(f"Loaded combo data with {combo_data.get('total_users', 0)} users")
        
        # Generate combo images
        logger.info("Generating combo images from existing suggestions...")
        image_results = combo_image_creator.process_combo_suggestions(combo_data)
        
        # Collect combo image S3 URLs
        combo_images = []
        for img_result in image_results.get('generated_images', []):
            if img_result.get('generation_success', False):
                s3_url = img_result.get('s3_url', '')
                if s3_url:
                    combo_images.append(s3_url)
        
        if not combo_images:
            raise HTTPException(status_code=500, detail="No combo images were generated successfully")
        
        logger.info(f"Generated {len(combo_images)} combo images")
        
        # Generate HTML email template
        logger.info("Generating HTML email template...")
        email_html = await generate_email_html(combo_images)
        
        # Save HTML to index.html
        try:
            with open('index.html', 'w', encoding='utf-8') as f:
                f.write(email_html)
            logger.info("📧 HTML template saved to index.html")
        except Exception as e:
            logger.warning(f"Failed to save HTML to index.html: {str(e)}")
        
        # Return only HTML
        response = UserEmailResponse(
            email_html=email_html
        )
        
        logger.info("✅ Test email generation completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in test email generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test email generation failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Fashion Search API",
        "version": "1.0.0",
        "description": "Analyze fashion images and find similar products using vector similarity search",
        "search_methods": {
            "knn": "Fast KNN-based vector search using OpenSearch's native KNN functionality",
            "simple": "Manual cosine similarity search (legacy method)"
        },
        "endpoints": {
            "POST /search/image": "Analyze image and find similar products (supports both KNN and simple search)",
            "POST /search/image-json": "Analyze base64 image and find similar products (supports both KNN and simple search)",
            "POST /search/multi-image-cluster": "Analyze multiple images (JSON), cluster them by similarity, and find similar products for each cluster",
            "POST /search/multi-image-cluster-form": "Analyze multiple images (form-data), cluster them by similarity, and find similar products for each cluster",
            "POST /search/text": "Search products by text query (supports both KNN and simple search)",
            "POST /search/product": "Search products by product data (supports both KNN and simple search)",
            "POST /generate-user-email": "Complete flow: Get user products → Analyze → Find similar → Generate combos → Create images → Generate HTML email",
            "POST /test-email-generation": "Fast test: Use existing combo_suggestions.json → Generate images → Generate HTML email",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get port from environment (Railway sets this)
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run(
        "fashion_search_api:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
