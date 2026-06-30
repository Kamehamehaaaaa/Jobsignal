# JobSignal

![CI](https://github.com/kamehamehaaaaa/JobSignal/actions/workflows/ci.yml/badge.svg)

A real-time job-hunt intelligence pipeline built on Kafka, Spark, and NLP. Monitors Gmail for job-related emails, classifies and enriches them with spaCy (or DistilBERT), and writes structured output to Google Sheets and local files — with full observability and CI/CD.

---

## Architecture

```
Gmail (IMAP + OAuth2)
        │
        ▼
Kafka Producer
        │
        ├── raw.emails.rejections
        └── raw.emails.applications
                │
                ▼
    Spark Structured Streaming
    (foreachBatch, 30s micro-batches)
                │
         spaCy NLP Enricher
         ├── Company extraction (NER)
         ├── Role title extraction (regex)
         └── Rejection type classifier
             (spaCy rules | DistilBERT zero-shot | DistilBERT fine-tuned)
                │
        ┌───────┴───────┐
        ▼               ▼
  Google Sheets    Local JSONL/CSV
  (Rejections)     (output/)
  (Applications)
```

**Dead-letter queues** (`dlq.emails.rejections`, `dlq.emails.applications`) capture any failed events for audit and replay.

---

## Project Structure

```
JobSignal/
├── config/
│   └── topics.py              # Kafka topic definitions, partition config, retention
├── schemas/
│   └── events.py              # Typed message schemas (frozen dataclasses + JSON serialisation)
├── producers/
│   ├── base_producer.py       # Shared Kafka producer logic, DLQ routing, delivery callbacks
│   ├── classifier.py          # Lightweight rule-based email gate (runs before Kafka)
│   └── gmail_producer.py      # Gmail IMAP + OAuth2 polling loop
├── consumer/
│   └── spark_consumer.py      # Spark Structured Streaming, foreachBatch, checkpointing
├── nlp/
│   ├── enricher.py            # spaCy NER + role/rejection enrichment, backend flag
│   └── distilbert_classifier.py  # DistilBERT zero-shot + fine-tune classifier
├── sinks/
│   ├── local_sink.py          # JSONL + CSV writer (development + backup)
│   └── sheets_sink.py         # Google Sheets API v4 writer
├── admin/
│   └── create_topics.py       # Idempotent Kafka topic creation script
├── tests/
│   ├── test_producers.py      # Classifier + schema unit tests (15 tests)
│   ├── test_consumer.py       # NLP enricher + local sink tests (24 tests)
│   └── test_distilbert.py     # DistilBERT flag dispatch tests (12 tests)
├── docker/
│   └── docker-compose.yml     # Kafka + Zookeeper + Kafka UI
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- Docker Desktop
- Java 17 (`brew install openjdk@17`)

### 2. Clone and install

```bash
git clone https://github.com/rohitbogulla/JobSignal.git
cd JobSignal
python -m venv .jobsignal
source .jobsignal/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure

```bash
cp .env.example .env
# Fill in GMAIL_USER_EMAIL, GMAIL_CREDENTIALS_PATH, KAFKA_BOOTSTRAP_SERVERS
```

Gmail OAuth2 setup:
1. Create a project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable the Gmail API
3. Create OAuth 2.0 credentials (Desktop App) → download as `secrets/gmail_credentials.json`
4. Add your Gmail address as a test user under OAuth consent screen

### 4. Start Kafka

```bash
docker compose -f docker/docker-compose.yml up -d

# Create topics
python -m admin.create_topics
```

### 5. Run the pipeline

```bash
# Terminal 1 — Spark consumer
python -m consumer.spark_consumer

# Terminal 2 — Gmail producer
python -m producers.gmail_producer
```

Structured output writes to `output/rejections.jsonl` and `output/applications.jsonl` automatically.

---

## NLP Backend

The rejection classifier supports three backends, switchable via environment variable — no code changes needed:

| `USE_DISTILBERT` | Backend | Accuracy | Setup |
|---|---|---|---|
| _(unset)_ | spaCy rules | ~90% | nothing extra |
| `zero_shot` | DistilBERT zero-shot | ~80–85% | `pip install transformers torch` |
| `finetuned` | Fine-tuned DistilBERT | ~93–96% | label data + run `--train` |

```bash
# Switch to zero-shot DistilBERT
USE_DISTILBERT=zero_shot python -m consumer.spark_consumer

# Fine-tune on labelled data (after collecting ~200+ examples)
python -m nlp.distilbert_classifier --train --data output/training_data.jsonl
```

---

## Kafka Topic Design

| Topic | Partitions | Retention | Purpose |
|---|---|---|---|
| `raw.emails.rejections` | 2 | 7 days | Rejection email events |
| `raw.emails.applications` | 2 | 7 days | Application confirmation events |
| `dlq.emails.rejections` | 1 | 30 days | Failed rejection events |
| `dlq.emails.applications` | 1 | 30 days | Failed application events |

Partition key = email source (`gmail` / `outlook`), guaranteeing per-source ordering.

---

## Running Tests

```bash
pytest tests/ -v
# 51 tests, ~8s
```

---

## Roadmap

- [ ] MLflow experiment tracking — classifier confidence, label distribution, model versioning
- [ ] GitHub Actions CI — test suite on every push
- [ ] Prometheus + Grafana — pipeline health dashboard
- [ ] Slack alerts — real-time rejection/application notifications
- [ ] Outlook producer — Microsoft Graph API (framework already built)
- [ ] DistilBERT fine-tuning — after 200+ labelled examples collected

---

## Tech Stack

| Layer | Technology |
|---|---|
| Message broker | Apache Kafka (Confluent 7.6) |
| Stream processing | Apache Spark 3.5 Structured Streaming |
| NLP | spaCy 3.8, DistilBERT (HuggingFace Transformers) |
| Gmail ingestion | IMAP + Google OAuth2 |
| Output | Google Sheets API v4, JSONL/CSV |
| Infrastructure | Docker, Docker Compose |
| Testing | pytest (51 tests) |
| Language | Python 3.11 |

### Key Components

| Component | Purpose |
|------------|----------|
| `config/topics.py` | Kafka topic definitions, partitioning strategy, and retention policies |
| `schemas/events.py` | Immutable event contracts shared across producers and consumers |
| `producers/` | Email ingestion pipelines for Gmail and Outlook |
| `admin/create_topics.py` | Idempotent Kafka topic provisioning |
| `tests/` | Unit tests for event schemas and classification logic |
| `docker/` | Local Kafka development environment |