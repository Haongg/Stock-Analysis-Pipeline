from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.flink.es_sink import (
    ElasticsearchSinkConfig,
    create_elasticsearch_client,
    ensure_elasticsearch_indices,
    setup_elasticsearch_lifecycle,
)


ELASTICSEARCH_DIR = Path(__file__).resolve().parent
RAW_INDEX_NAME = "stock-raw-ohlcv"
RAW_MAPPING_PATH = ELASTICSEARCH_DIR / "stock_raw_ohlcv_mapping.json"


def create_raw_index_if_missing(es_client) -> bool:
    """Create the raw OHLCV bootstrap index if it is not present."""
    if es_client.indices.exists(index=RAW_INDEX_NAME):
        print(f"Index '{RAW_INDEX_NAME}' already exists. Skipping creation.")
        return False

    with RAW_MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
        mapping = json.load(mapping_file)

    es_client.indices.create(index=RAW_INDEX_NAME, body=mapping)
    print(f"Successfully created index '{RAW_INDEX_NAME}'.")
    return True


def main() -> None:
    """Create stock analysis Elasticsearch indices."""
    config = ElasticsearchSinkConfig()
    es_client = create_elasticsearch_client(config)

    create_raw_index_if_missing(es_client)
    lifecycle = setup_elasticsearch_lifecycle(es_client, config=config)
    if lifecycle["ilm_enabled"]:
        print(f"ILM policy '{lifecycle['policy']}' and templates verified.")
    results = ensure_elasticsearch_indices(es_client, config=config)
    for index_name, created in results.items():
        action = "created" if created else "verified"
        print(f"Index '{index_name}' {action}.")


if __name__ == "__main__":
    main()
