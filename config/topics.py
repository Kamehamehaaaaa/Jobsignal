from dataclasses import dataclass

TOPIC_REJECTIONS = "raw.emails.rejections"
TOPIC_APPLICATIONS = "raw.emails.applications"
TOPIC_DLQ_REJECTIONS   = "dlq.emails.rejections"
TOPIC_DLQ_APPLICATIONS = "dlq.emails.applications"

PARTITION_KEY_GMAIL = b"gmail"
PARTITION_KEY_OUTLOOK = b"outlook"

@dataclass
class TopicSpec:
    name: str
    num_partitions: int
    replication_factor: int
    config: dict

TOPIC_SPECS : list[TopicSpec] = [
    TopicSpec(
        name = TOPIC_REJECTIONS,
        num_partitions = 2,     # for parallelism, throughput and ordering guarantees. 2 consumers can consume.
        replication_factor = 1,
        config = {
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),   # 7 days
            "cleanup.policy": "delete",
            "compression.type": "lz4",
        }
    ),
    TopicSpec(
        name=TOPIC_APPLICATIONS,
        num_partitions=2,
        replication_factor=1,
        config={
            "retention.ms": str(7 * 24 * 60 * 60 * 1000),
            "cleanup.policy": "delete",
            "compression.type": "lz4",
        },
    ),
    TopicSpec(
        name=TOPIC_DLQ_REJECTIONS,
        num_partitions=1,
        replication_factor=1,
        config={
            "retention.ms": str(30 * 24 * 60 * 60 * 1000),  # 30 days — keep failures longer
            "cleanup.policy": "delete",
        },
    ),
    TopicSpec(
        name=TOPIC_DLQ_APPLICATIONS,
        num_partitions=1,
        replication_factor=1,
        config={
            "retention.ms": str(30 * 24 * 60 * 60 * 1000),
            "cleanup.policy": "delete",
        },
    ),
]