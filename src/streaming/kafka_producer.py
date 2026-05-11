"""
Kafka producer utilities for stock data ingestion.

This module initializes a producer from environment variables and sends
JSON payloads encoded as UTF-8 bytes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from kafka import KafkaProducer


def create_kafka_producer(
    bootstrap_servers: Optional[str] = None,
    client_id: Optional[str] = None,
) -> KafkaProducer:
    """Create a Kafka producer configured for JSON values and string keys."""
    servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092,localhost:29092,localhost:39092",)
    kafka_client_id = client_id or os.getenv("KAFKA_CLIENT_ID", "stock-producer")

    return KafkaProducer(
        bootstrap_servers=[s.strip() for s in servers.split(",") if s.strip()], # server ports (internal/external)
        client_id=kafka_client_id, # name of producer
        acks="all", # only produce successfully if all brokers have to have replicas 
        retries=5,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def publish_json(
    producer: KafkaProducer,
    key: str,
    value: Dict[str, Any],
    topic: Optional[str] = None,
) -> None:
    """Publish one JSON record and wait for broker acknowledgement."""
    target_topic = topic or os.getenv("KAFKA_TOPIC_RAW", "stock.raw.ohlcv")
    # key (ticker) and value (OHLCV) is postition match with key_serializer and value_serializer
    future = producer.send(topic=target_topic, key=key, value=value)
    future.get(timeout=30)


def close_producer(producer: KafkaProducer) -> None:
    """Flush pending messages and close producer cleanly."""
    producer.flush(timeout=10)
    producer.close()
