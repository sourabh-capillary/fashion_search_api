#!/usr/bin/env python3
"""
Simple Vector Search
A working vector search implementation that doesn't rely on k-NN or script_score.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from aws_requests_auth.aws_auth import AWSRequestsAuth
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleVectorSearch:
    def __init__(self, 
                 opensearch_endpoint: str,
                 opensearch_index: str = "products",
                 aws_region: str = "us-east-1",
                 openai_api_key: str = None):
        """Initialize the Simple Vector Search."""
        self.opensearch_endpoint = opensearch_endpoint
        self.opensearch_index = opensearch_index
        self.aws_region = aws_region
        
        # Initialize OpenAI client
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("OpenAI API key is required.")
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Initialize AWS session
        self.aws_session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=aws_region
        )
        
        # Initialize OpenSearch client
        self.opensearch_client = self._create_opensearch_client()
        
    def _create_opensearch_client(self) -> OpenSearch:
        """Create and configure OpenSearch client with AWS authentication."""
        try:
            # Get AWS credentials
            credentials = self.aws_session.get_credentials()
            
            # Create AWS auth
            awsauth = AWSRequestsAuth(
                aws_access_key=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_token=credentials.token,
                aws_host=self.opensearch_endpoint.replace('https://', ''),
                aws_region=self.aws_region,
                aws_service='es'
            )
            
            # Create OpenSearch client
            client = OpenSearch(
                hosts=[{'host': self.opensearch_endpoint.replace('https://', ''), 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=60
            )
            
            logger.info("OpenSearch client initialized successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create OpenSearch client: {str(e)}")
            raise
    
    def generate_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embedding for the given text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_similar_products(self, 
                              query_text: str, 
                              size: int = 10) -> List[Dict[str, Any]]:
        """Search for similar products using vector similarity."""
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query_text)
            
            # Fetch all documents
            logger.info("Fetching all documents...")
            response = self.opensearch_client.search(
                index=self.opensearch_index,
                body={
                    "query": {"match_all": {}},
                    "size": 1000  # Get all documents
                }
            )
            
            documents = response['hits']['hits']
            logger.info(f"Found {len(documents)} documents")
            
            # Calculate similarities
            similarities = []
            for doc in documents:
                source = doc['_source']
                
                if 'embeddings' in source and isinstance(source['embeddings'], list):
                    # Calculate cosine similarity
                    similarity = self.cosine_similarity(query_embedding, source['embeddings'])
                    
                    similarities.append({
                        'doc': doc,
                        'similarity': similarity
                    })
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top results
            results = []
            for item in similarities[:size]:
                doc = item['doc']
                result = {
                    'product_id': doc['_source'].get('product_id'),
                    's3_url': doc['_source'].get('s3_url'),
                    'metadata': doc['_source'].get('metadata', {}),
                    'similarity_score': item['similarity']
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar products")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar products: {str(e)}")
            raise
    
    def search_by_product_data(self, 
                             product_data: Dict[str, Any], 
                             size: int = 10) -> List[Dict[str, Any]]:
        """Search for similar products using product data."""
        try:
            # Extract metadata
            metadata = product_data.get('metadata', {})
            
            # Create text from metadata
            text_parts = []
            for key, value in metadata.items():
                if value and value != "None":
                    readable_key = key.replace('_', ' ').title()
                    text_parts.append(f"{readable_key}: {value}")
            
            metadata_text = " | ".join(text_parts)
            logger.info(f"Metadata text: {metadata_text}")
            
            # Search using the metadata text
            return self.search_similar_products(metadata_text, size)
            
        except Exception as e:
            logger.error(f"Failed to search by product data: {str(e)}")
            raise


def main():
    """Main function to demonstrate simple vector search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Vector Search for OpenSearch Products")
    parser.add_argument("--endpoint", "-e", required=True,
                       help="AWS OpenSearch domain endpoint")
    parser.add_argument("--index", "-i", default="products",
                       help="OpenSearch index name (default: products)")
    parser.add_argument("--region", "-r", default="us-east-1",
                       help="AWS region (default: us-east-1)")
    parser.add_argument("--query", "-q", 
                       help="Text query for similarity search")
    parser.add_argument("--product-data", "-p", 
                       help="Product data as JSON string")
    parser.add_argument("--size", "-s", type=int, default=5,
                       help="Number of results to return (default: 5)")
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        logger.info("Initializing Simple Vector Search...")
        searcher = SimpleVectorSearch(
            opensearch_endpoint=args.endpoint,
            opensearch_index=args.index,
            aws_region=args.region
        )
        
        results = []
        
        if args.product_data:
            # Use product data
            product_data = json.loads(args.product_data)
            logger.info(f"Searching with product data: {product_data.get('product_id', 'Unknown')}")
            results = searcher.search_by_product_data(product_data, args.size)
            
        elif args.query:
            # Use text query
            logger.info(f"Searching with query: '{args.query}'")
            results = searcher.search_similar_products(args.query, args.size)
            
        else:
            logger.error("Please provide either --query or --product-data")
            return
        
        # Display results
        if results:
            print("\n" + "=" * 80)
            print("SEARCH RESULTS")
            print("=" * 80)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Product ID: {result['product_id']}")
                print(f"   S3 URL: {result['s3_url']}")
                print(f"   Similarity Score: {result['similarity_score']:.4f}")
                
                print("   Metadata:")
                for key, value in result['metadata'].items():
                    if value and value != "None":
                        print(f"     {key}: {value}")
        else:
            print("No results found.")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Make sure your AWS credentials are correct")
        logger.error("2. Check that your OpenSearch endpoint is accessible")
        logger.error("3. Ensure documents have been processed with embeddings")
        logger.error("4. Verify your OpenAI API key is valid")


if __name__ == "__main__":
    main()
