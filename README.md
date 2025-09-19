# Fashion Search API - Deployment Package

## Overview
This is a FastAPI application that generates personalized fashion combo emails with AI-generated images.

## Features
- **Persona Analysis**: Analyzes fashion images using GPT-4 Vision
- **Similar Product Search**: Finds similar products using OpenSearch vector search
- **Combo Generation**: Creates product combinations using LLM
- **Image Generation**: Generates combo images using GPT-Image-1
- **S3 Integration**: Uploads images to S3 and uses S3 URLs in emails
- **HTML Email Templates**: Generates professional email templates

## API Endpoints

### Main Endpoints
- `POST /generate-user-email` - Complete flow for a specific user
- `POST /test-email-generation` - Fast test using existing combo data
- `GET /health` - Health check
- `GET /docs` - API documentation

### Search Endpoints
- `POST /search/image` - Analyze image and find similar products
- `POST /search/text` - Search products by text query
- `POST /search/product` - Search products by product data

## Required Environment Variables

```bash
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Anthropic API (optional, has fallback)
ANTHROPIC_API_KEY=your_anthropic_api_key

# AWS Configuration
AWS_REGION=us-east-1
S3_BUCKET_NAME=hacka-image-bucket

# OpenSearch Configuration
OPENSEARCH_ENDPOINT=your_opensearch_endpoint
OPENSEARCH_INDEX=products
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your actual values
```

3. Run the application:
```bash
python fashion_search_api.py
```

## Deployment Options

### Option 1: Local Development
```bash
python fashion_search_api.py
```

### Option 2: Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "fashion_search_api.py"]
```

### Option 3: AWS Lambda
Use the `lambda_customer_analysis.py` for serverless deployment.

## File Structure

```
deployment/
├── fashion_search_api.py      # Main FastAPI application
├── persona_analyzer.py        # Image analysis using GPT-4 Vision
├── simple_vector_search.py    # Vector search implementation
├── knn_vector_search.py       # KNN vector search
├── combo_generator.py         # Product combination generation
├── combo_image_creator.py     # Image generation and S3 upload
├── lambda_customer_analysis.py # Lambda function
├── customer_data.csv          # Sample customer data
├── combo_suggestions.json     # Sample combo data
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables
└── README.md                 # This file
```

## Usage Examples

### Generate Email for User
```bash
curl -X POST "http://localhost:8000/generate-user-email" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "46382",
    "max_combos": 2,
    "combo_size": 3
  }'
```

### Test Email Generation
```bash
curl -X POST "http://localhost:8000/test-email-generation" \
  -H "Content-Type: application/json" \
  -d '{"use_existing_combos": true}'
```

## Notes
- Images are automatically uploaded to S3
- HTML templates use S3 URLs for email compatibility
- Random filenames are generated for combo images
- Fallback mechanisms ensure API reliability
