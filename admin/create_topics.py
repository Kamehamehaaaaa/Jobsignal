"""
JobSignal — Kafka topic admin script.

Creates all required topics idempotently (safe to run multiple times).
Existing topics with the correct config are left untouched.

Usage:
    python -m jobsignal.admin.create_topics

Environment variables (or .env file):
    KAFKA_BOOTSTRAP_SERVERS   default: localhost:9092
"""

from __future__ import annotations
import logging
import os

from dotenv import load_dotenv
from confluent_kafka.admin import AdminClient, NewTopic, ConfigResource, ConfigSource

from config.topics import TOPIC_SPECS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("create_topics")


def create_topics(bootstrap_servers: str) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})

    # List existing topics
    metadata = admin.list_topics(timeout=15)
    existing = set(metadata.topics.keys())
    logger.info("Existing topics: %s", sorted(existing))

    to_create = [
        NewTopic(
            spec.name,
            num_partitions=spec.num_partitions,
            replication_factor=spec.replication_factor,
            config=spec.config,
        )
        for spec in TOPIC_SPECS
        if spec.name not in existing
    ]

    if not to_create:
        logger.info("All %d topics already exist. Nothing to do.", len(TOPIC_SPECS))
        return

    futures = admin.create_topics(to_create, validate_only=False)
    for name, fut in futures.items():
        try:
            fut.result()
            logger.info("✓ Created topic: %s", name)
        except Exception as exc:
            logger.error("✗ Failed to create topic %s: %s", name, exc)

    # Print a summary of all topics after creation
    metadata = admin.list_topics(timeout=10)
    logger.info(
        "Topics now present: %s",
        sorted(t for t in metadata.topics if not t.startswith("__")),
    )


if __name__ == "__main__":
    servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    logger.info("Connecting to Kafka at %s", servers)
    create_topics(servers)