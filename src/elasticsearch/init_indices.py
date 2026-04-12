import json
import os
from elasticsearch import Elasticsearch

# Elasticsearch configuration
ES_HOST = os.getenv('ES_HOST', 'localhost') # localhost is default with http, not https
ES_PORT = os.getenv('ES_PORT', 9200)
ES_USER = os.getenv('ES_USER', None)
ES_PASSWORD = os.getenv('ES_PASSWORD', None)

def create_elasticsearch_client():
    """Create and return Elasticsearch client."""
    return Elasticsearch(
        hosts=[f"http://{ES_HOST}:{ES_PORT}"],
        http_auth=(ES_USER, ES_PASSWORD) if ES_USER and ES_PASSWORD else None,
        verify_certs=False
    )

def create_index_from_mapping(es_client, index_name, mapping_file_path):
    """Create Elasticsearch index from mapping file."""
    try:
        with open(mapping_file_path, 'r') as f:
            mapping = json.load(f)

        if es_client.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists. Skipping creation.")
            return False

        response = es_client.indices.create(index=index_name, body=mapping)
        print(f"Successfully created index '{index_name}': {response}")
        return True

    except Exception as e:
        print(f"Error creating index '{index_name}': {e}")
        return False

def main():
    """Main function to create all stock analysis indices."""
    es_client = create_elasticsearch_client()

    # Define index names and their mapping files
    indices = {
        'stock-raw-ohlcv': 'src/elasticsearch/stock_raw_ohlcv_mapping.json',
        'stock-engineered-features': 'src/elasticsearch/stock_engineered_features_mapping.json',
        'stock-predictions': 'src/elasticsearch/stock_predictions_mapping.json'
    }

    # Create each index
    for index_name, mapping_file in indices.items():
        mapping_path = os.path.join(os.getcwd(), mapping_file)
        if os.path.exists(mapping_path):
            create_index_from_mapping(es_client, index_name, mapping_path)
        else:
            print(f"Mapping file not found: {mapping_path}")

if __name__ == "__main__":
    main()