#!/usr/bin/env python3
"""
KNN Vector Search for OpenSearch
Proper implementation using OpenSearch's built-in KNN functionality.
"""

import os
import json
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


class KNNVectorSearch:
    def __init__(self, 
                 opensearch_endpoint: str,
                 opensearch_index: str = "products",
                 aws_region: str = "us-east-1",
                 openai_api_key: str = None):
        """Initialize the KNN Vector Search."""
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
    
    def create_metadata_text(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata dictionary to a text string for embedding."""
        text_parts = []
        
        for key, value in metadata.items():
            if value and value != "None":
                # Convert camelCase to readable format
                readable_key = key.replace('_', ' ').title()
                text_parts.append(f"{readable_key}: {value}")
        
        return " | ".join(text_parts)
    
    def search_similar_products_knn(self, 
                                   query_text: str, 
                                   size: int = 10,
                                   k: int = 100) -> List[Dict[str, Any]]:
        """
        Search for similar products using OpenSearch KNN functionality.
        
        Args:
            query_text: Text query for similarity search
            size: Number of results to return
            k: Number of nearest neighbors to consider (should be >= size)
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query_text)
            
            # KNN search query
            search_body = {
                "size": size,
                "query": {
                    "knn": {
                        "embeddings": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                }
            }
            
            logger.info(f"Performing KNN search with k={k}, size={size}")
            response = self.opensearch_client.search(
                index=self.opensearch_index,
                body=search_body
            )
            
            hits = response['hits']['hits']
            logger.info(f"Found {len(hits)} similar products using KNN")
            
            # Process results
            results = []
            for hit in hits:
                source = hit['_source']
                result = {
                    'product_id': source.get('product_id'),
                    's3_url': source.get('s3_url'),
                    'metadata': source.get('metadata', {}),
                    'similarity_score': hit.get('_score', 0.0)  # KNN score from OpenSearch
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar products using KNN: {str(e)}")
            raise
    
    def search_by_product_data_knn(self, 
                                  product_data: Dict[str, Any], 
                                  size: int = 10,
                                  k: int = 100) -> List[Dict[str, Any]]:
        """Search for similar products using product data with KNN."""
        try:
            # Extract metadata
            metadata = product_data.get('metadata', {})
            
            # Create text from metadata
            metadata_text = self.create_metadata_text(metadata)
            logger.info(f"Metadata text: {metadata_text}")
            
            # Search using the metadata text with KNN
            return self.search_similar_products_knn(metadata_text, size, k)
            
        except Exception as e:
            logger.error(f"Failed to search by product data using KNN: {str(e)}")
            raise
    
    def hybrid_search(self, 
                     query_text: str, 
                     size: int = 10,
                     k: int = 100,
                     text_weight: float = 0.3,
                     vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search combining KNN vector search with text search.
        
        Args:
            query_text: Text query for search
            size: Number of results to return
            k: Number of nearest neighbors for KNN
            text_weight: Weight for text search (0.0 to 1.0)
            vector_weight: Weight for vector search (0.0 to 1.0)
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query_text)
            
            # Hybrid search query
            search_body = {
                "size": size,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "embeddings": {
                                        "vector": query_embedding,
                                        "k": k
                                    },
                                    "boost": vector_weight
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query_text,
                                    "fields": ["metadata.*"],
                                    "boost": text_weight
                                }
                            }
                        ]
                    }
                }
            }
            
            logger.info(f"Performing hybrid search with vector_weight={vector_weight}, text_weight={text_weight}")
            response = self.opensearch_client.search(
                index=self.opensearch_index,
                body=search_body
            )
            
            hits = response['hits']['hits']
            logger.info(f"Found {len(hits)} results using hybrid search")
            
            # Process results
            results = []
            for hit in hits:
                source = hit['_source']
                result = {
                    'product_id': source.get('product_id'),
                    's3_url': source.get('s3_url'),
                    'metadata': source.get('metadata', {}),
                    'similarity_score': hit.get('_score', 0.0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {str(e)}")
            raise
    
    def test_knn_functionality(self) -> bool:
        """Test if KNN functionality is working properly."""
        try:
            # Test with a simple query
            test_query = "men casual wear"
            results = self.search_similar_products_knn(test_query, size=1, k=10)
            
            if results:
                logger.info("✅ KNN functionality is working!")
                logger.info(f"Test query '{test_query}' returned {len(results)} results")
                return True
            else:
                logger.warning("⚠️ KNN search returned no results - check if embeddings exist")
                return False
                
        except Exception as e:
            logger.error(f"❌ KNN functionality test failed: {str(e)}")
            return False
    
    def get_index_mapping(self) -> Dict[str, Any]:
        """Get the current index mapping to verify KNN configuration."""
        try:
            response = self.opensearch_client.indices.get_mapping(index=self.opensearch_index)
            return response
        except Exception as e:
            logger.error(f"Failed to get index mapping: {str(e)}")
            return {}


def main():
    """Main function to demonstrate KNN vector search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KNN Vector Search for OpenSearch Products")
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
    parser.add_argument("--k", "-k", type=int, default=100,
                       help="Number of nearest neighbors to consider (default: 100)")
    parser.add_argument("--hybrid", action="store_true",
                       help="Use hybrid search (KNN + text search)")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Test KNN functionality")
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        logger.info("Initializing KNN Vector Search...")
        searcher = KNNVectorSearch(
            opensearch_endpoint=args.endpoint,
            opensearch_index=args.index,
            aws_region=args.region
        )
        
        if args.test:
            # Test KNN functionality
            logger.info("Testing KNN functionality...")
            if searcher.test_knn_functionality():
                logger.info("✅ KNN is working correctly!")
            else:
                logger.error("❌ KNN is not working properly")
            return
        
        results = []
        
        if args.product_data:
            # Use product data
            product_data = json.loads(args.product_data)
            logger.info(f"Searching with product data: {product_data.get('product_id', 'Unknown')}")
            
            if args.hybrid:
                metadata_text = searcher.create_metadata_text(product_data.get('metadata', {}))
                results = searcher.hybrid_search(metadata_text, args.size, args.k)
            else:
                results = searcher.search_by_product_data_knn(product_data, args.size, args.k)
            
        elif args.query:
            # Use text query
            logger.info(f"Searching with query: '{args.query}'")
            
            if args.hybrid:
                results = searcher.hybrid_search(args.query, args.size, args.k)
            else:
                results = searcher.search_similar_products_knn(args.query, args.size, args.k)
            
        else:
            logger.error("Please provide either --query or --product-data")
            return
        
        # Display results
        if results:
            print("\n" + "=" * 80)
            print("KNN SEARCH RESULTS")
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
        logger.error("5. Check that your index has KNN enabled and embeddings field is properly mapped")


if __name__ == "__main__":
    main()
