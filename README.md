## Architecture

```text
jobsignal/
├── config/
│   └── topics.py
│
├── schemas/
│   └── events.py
│
├── producers/
│   ├── base_producer.py
│   ├── classifier.py
│   ├── gmail_producer.py
│   └── outlook_producer.py
│
├── admin/
│   └── create_topics.py
│
├── tests/
│   └── test_producers.py
│
├── docker/
│   └── docker-compose.yml
│
└── .env.example
```

### Key Components

| Component | Purpose |
|------------|----------|
| `config/topics.py` | Kafka topic definitions, partitioning strategy, and retention policies |
| `schemas/events.py` | Immutable event contracts shared across producers and consumers |
| `producers/` | Email ingestion pipelines for Gmail and Outlook |
| `admin/create_topics.py` | Idempotent Kafka topic provisioning |
| `tests/` | Unit tests for event schemas and classification logic |
| `docker/` | Local Kafka development environment |
