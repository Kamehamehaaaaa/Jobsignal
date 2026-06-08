from __future__ import annotations
import logging
from typing import Optional
 
from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
 
from config.topics import TopicSpec, TOPIC_DLQ_REJECTIONS, TOPIC_DLQ_APPLICATIONS
from schemas.events import make_dlq_event
 
logger = logging.getLogger(__name__)


def _delivery_callback(err, msg):
    if err:
        logger.error(
            "Delivery failed | topic=%s partition=%d offset=%s key=%s | %s",
            msg.topic(), msg.partition(), msg.offset(), msg.key(), err,
        )
    else:
        logger.debug(
            "Delivered | topic=%s partition=%d offset=%d key=%s",
            msg.topic(), msg.partition(), msg.offset(), msg.key(),
        )

class BaseProducer:
    def __init__(
        self,
        kafka_config: dict,
        producer_name: str,
        dlq_topic: str,
    ):
        self.producer_name = producer_name
        self.dlq_topic = dlq_topic
        self._producer = Producer(
            {
                # Defaults that make sense for all producers in this project.
                # Callers can override by passing the same key in kafka_config.
                "acks": "all",             # wait for all in-sync replicas
                "retries": 5,
                "retry.backoff.ms": 300,
                "compression.type": "lz4",
                "linger.ms": 20,           # small batching window for throughput
                **kafka_config,
            }
        )
        logger.info("Producer initialised | name=%s dlq=%s", producer_name, dlq_topic)

    def send(self, topic: str, key: bytes, value: bytes, headers: Optional[dict] = None):
        """
        Produce a single message.  On KafkaException, the raw payload is
        forwarded to the DLQ so no event is silently lost.
        """
        kafka_headers = list(headers.items()) if headers else []
        try:
            self._producer.produce(
                topic=topic,
                key=key,
                value=value,
                headers=kafka_headers,
                on_delivery=_delivery_callback
            )
            self._producer.poll(0)
        except KafkaException as err:
            logger.error(
                "Produce failed, routing to DLQ | topic=%s | %s", topic, err
            )
            self._route_to_dlq(topic=topic, raw_payload=value, reason=str(err))
        except BufferError:
            logger.warning("Producer queue full, flushing then retrying once")
            self._producer.flush(timeout=10)
            self.send(topic, key, value, headers)   # only one retry

    
    def _route_to_dlq(self, topic: str, raw_payload: bytes, reason: str):
        dlq_event = make_dlq_event(
            original_topic=topic,
            failure_reason=reason,
            raw_payload=raw_payload.decode("utf-8", errors="replace"),
            producer_name=self.producer_name
        )

        try:
            self._producer.produce(
                topic=self.dlq_topic,
                key=b"dlq",
                value=dlq_event.to_json(),
            )

            self._producer.poll(0)
            logger.info("DLQ event written | dlq_topic=%s", self.dlq_topic)
        
        except Exception as dlq_exc:
            logger.critical(
                "DLQ write failed — payload logged here for manual recovery | "
                "dlq_topic=%s | reason=%s | payload=%s | dlq_error=%s",
                self.dlq_topic,
                reason,
                raw_payload.decode("utf-8", errors="replace"),
                dlq_exc,
            )

    def flush(self, timeout: float = 30.0) -> None:
        """Block until all outstanding messages are delivered."""
        remaining = self._producer.flush(timeout=timeout)
        if remaining > 0:
            logger.warning(
                "%d message(s) not delivered after %.1fs flush", remaining, timeout
            )
 
    def __enter__(self) -> "BaseProducer":
        return self
 
    def __exit__(self, *_) -> None:
        self.flush()
        logger.info("Producer flushed and closed | name=%s", self.producer_name)


def ensure_topics(bootstrap_servers: str, topic_specs: list[TopicSpec]) -> None:
    """
    Idempotent topic creation.  Safe to call on every startup — existing
    topics are left untouched; only missing ones are created.
    """
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    existing = set(admin.list_topics(timeout=10).topics.keys())
 
    to_create = [
        NewTopic(
            spec.name,
            num_partitions=spec.num_partitions,
            replication_factor=spec.replication_factor,
            config=spec.config,
        )
        for spec in topic_specs
        if spec.name not in existing
    ]
 
    if not to_create:
        logger.info("All topics already exist, nothing to create")
        return
 
    futures = admin.create_topics(to_create)
    for topic_name, future in futures.items():
        try:
            future.result()
            logger.info("Created topic: %s", topic_name)
        except Exception as exc:
            logger.error("Failed to create topic %s: %s", topic_name, exc)

 