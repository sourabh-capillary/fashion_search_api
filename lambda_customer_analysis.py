#!/usr/bin/env python3
"""
AWS Lambda function for customer product analysis and similar product recommendation.
Processes CSV file with user products, analyzes each product's persona using S3 URLs,
finds similar products, and stores results in JSON format.
"""

import json
import csv
import os
import logging
from typing import List, Dict, Any, Optional
from io import StringIO
import boto3
from botocore.exceptions import ClientError

# Import your existing modules
from persona_analyzer import PersonaAnalyzer
from knn_vector_search import KNNVectorSearch

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for initialized components
persona_analyzer = None
knn_vector_search = None

def initialize_components():
    """Initialize the PersonaAnalyzer and KNNVectorSearch components."""
    global persona_analyzer, knn_vector_search
    
    try:
        # Initialize Persona Analyzer
        persona_analyzer = PersonaAnalyzer()
        logger.info("✅ Persona Analyzer initialized")
        
        # Initialize KNN Vector Search
        opensearch_endpoint = os.getenv("OPENSEARCH_ENDPOINT")
        if not opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT environment variable is required")
        
        opensearch_index = os.getenv("OPENSEARCH_INDEX", "products")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        
        knn_vector_search = KNNVectorSearch(
            opensearch_endpoint=opensearch_endpoint,
            opensearch_index=opensearch_index,
            aws_region=aws_region
        )
        logger.info("✅ KNN Vector Search initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {str(e)}")
        raise

def process_csv_data(csv_content: str) -> List[Dict[str, str]]:
    """
    Process CSV content and return list of user products.
    
    Args:
        csv_content: CSV content as string
        
    Returns:
        List of dictionaries with user_id, product_id, s3_url
    """
    products = []
    
    try:
        # Parse CSV content
        csv_reader = csv.DictReader(StringIO(csv_content))
        
        for row in csv_reader:
            if 'user_id' in row and 'product_id' in row and 's3_url' in row:
                products.append({
                    'user_id': row['user_id'].strip(),
                    'product_id': row['product_id'].strip(),
                    's3_url': row['s3_url'].strip()
                })
            else:
                logger.warning(f"Skipping invalid row: {row}")
        
        logger.info(f"Processed {len(products)} products from CSV")
        return products
        
    except Exception as e:
        logger.error(f"Failed to process CSV data: {str(e)}")
        raise

def analyze_product_persona(s3_url: str, product_id: str) -> Dict[str, Any]:
    """
    Analyze product persona from S3 URL.
    
    Args:
        s3_url: S3 URL of the product image
        product_id: Product ID
        
    Returns:
        Dictionary containing persona analysis results
    """
    try:
        logger.info(f"Analyzing persona for product {product_id} from {s3_url}")
        
        # Analyze image from URL
        analysis_result = persona_analyzer.analyze_image_from_url(s3_url)
        
        if 'error' in analysis_result:
            logger.error(f"Failed to analyze product {product_id}: {analysis_result['error']}")
            return {
                'product_id': product_id,
                's3_url': s3_url,
                'error': analysis_result['error'],
                'metadata': {}
            }
        
        # Extract metadata (exclude error, image_url, filename if present)
        metadata = {k: v for k, v in analysis_result.items() 
                   if k not in ['error', 'image_url', 'filename']}
        
        return {
            'product_id': product_id,
            's3_url': s3_url,
            'metadata': metadata,
            'analysis_success': True
        }
        
    except Exception as e:
        logger.error(f"Error analyzing product {product_id}: {str(e)}")
        return {
            'product_id': product_id,
            's3_url': s3_url,
            'error': str(e),
            'metadata': {},
            'analysis_success': False
        }

def find_similar_products(product_data: Dict[str, Any], max_results: int = 2) -> List[Dict[str, Any]]:
    """
    Find similar products for a given product.
    
    Args:
        product_data: Product data with metadata
        max_results: Maximum number of similar products to return
        
    Returns:
        List of similar products
    """
    try:
        if not product_data.get('analysis_success', False):
            logger.warning(f"Skipping similar product search for {product_data['product_id']} - analysis failed")
            return []
        
        logger.info(f"Finding similar products for {product_data['product_id']}")
        
        # Search for similar products using KNN
        similar_products = knn_vector_search.search_by_product_data_knn(
            product_data, 
            max_results
        )
        
        logger.info(f"Found {len(similar_products)} similar products for {product_data['product_id']}")
        return similar_products
        
    except Exception as e:
        logger.error(f"Error finding similar products for {product_data['product_id']}: {str(e)}")
        return []

def process_user_products(user_products: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Process all products for a user and find similar products.
    
    Args:
        user_products: List of user products with user_id, product_id, s3_url
        
    Returns:
        Dictionary containing processed results
    """
    results = {
        'user_id': user_products[0]['user_id'] if user_products else 'unknown',
        'total_products': len(user_products),
        'processed_products': [],
        'processing_summary': {
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_similar_products_found': 0
        }
    }
    
    for product in user_products:
        user_id = product['user_id']
        product_id = product['product_id']
        s3_url = product['s3_url']
        
        logger.info(f"Processing product {product_id} for user {user_id}")
        
        # Analyze product persona
        analysis_result = analyze_product_persona(s3_url, product_id)
        
        # Find similar products if analysis was successful
        similar_products = []
        if analysis_result.get('analysis_success', False):
            similar_products = find_similar_products(analysis_result, max_results=2)
            results['processing_summary']['successful_analyses'] += 1
            results['processing_summary']['total_similar_products_found'] += len(similar_products)
        else:
            results['processing_summary']['failed_analyses'] += 1
        
        # Store results
        product_result = {
            'original_product': {
                'product_id': product_id,
                's3_url': s3_url,
                'user_id': user_id
            },
            'persona_analysis': analysis_result,
            'similar_products': similar_products
        }
        
        results['processed_products'].append(product_result)
    
    return results

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event containing CSV data or S3 reference
        context: Lambda context
        
    Returns:
        Dictionary containing processing results
    """
    try:
        # Initialize components
        initialize_components()
        
        # Extract CSV data from event
        csv_content = None
        
        if 'csv_data' in event:
            # CSV data provided directly in event
            csv_content = event['csv_data']
        elif 's3_bucket' in event and 's3_key' in event:
            # CSV data stored in S3
            s3_client = boto3.client('s3')
            try:
                response = s3_client.get_object(
                    Bucket=event['s3_bucket'],
                    Key=event['s3_key']
                )
                csv_content = response['Body'].read().decode('utf-8')
            except ClientError as e:
                logger.error(f"Failed to read CSV from S3: {str(e)}")
                raise
        else:
            raise ValueError("Either 'csv_data' or 's3_bucket'/'s3_key' must be provided in event")
        
        # Process CSV data
        user_products = process_csv_data(csv_content)
        
        if not user_products:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'No valid products found in CSV data',
                    'message': 'CSV must contain user_id, product_id, and s3_url columns'
                })
            }
        
        # Group products by user_id
        user_groups = {}
        for product in user_products:
            user_id = product['user_id']
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(product)
        
        # Process each user's products
        all_results = []
        for user_id, products in user_groups.items():
            logger.info(f"Processing {len(products)} products for user {user_id}")
            user_results = process_user_products(products)
            all_results.append(user_results)
        
        # Create final response
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Customer product analysis completed successfully',
                'total_users': len(user_groups),
                'total_products_processed': len(user_products),
                'results': all_results,
                'summary': {
                    'total_users': len(user_groups),
                    'total_products': len(user_products),
                    'successful_analyses': sum(r['processing_summary']['successful_analyses'] for r in all_results),
                    'failed_analyses': sum(r['processing_summary']['failed_analyses'] for r in all_results),
                    'total_similar_products_found': sum(r['processing_summary']['total_similar_products_found'] for r in all_results)
                }
            }, indent=2)
        }
        # json.dumps(similar_products_results = response, indent=2)
        
        logger.info("✅ Customer product analysis completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Lambda execution failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def main():
    """
    Main function for local testing.
    """
    # Example usage for local testing
    sample_csv = """user_id,product_id,s3_url
46382,8909230826683,https://hacka-image-bucket.s3.us-east-1.amazonaws.com/customer_products/8909230826683.png
46382,8905807281534,https://hacka-image-bucket.s3.us-east-1.amazonaws.com/customer_products/8905807281534.png
46382,8909203012921,https://hacka-image-bucket.s3.us-east-1.amazonaws.com/customer_products/8909203012921.png
46382,8909230741184,https://hacka-image-bucket.s3.us-east-1.amazonaws.com/customer_products/8909230741184.png"""
    
    # Create test event
    test_event = {
        'csv_data': sample_csv
    }
    
    # Test the lambda function
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
