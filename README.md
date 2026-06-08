jobsignal/
├── config/topics.py          — topic names, partition counts, retention config
├── schemas/events.py         — canonical message schemas (frozen dataclasses)
├── producers/
│   ├── base_producer.py      — shared Kafka logic (DLQ routing, delivery callbacks)
│   ├── classifier.py         — lightweight pre-NLP gate (runs in producer)
│   ├── gmail_producer.py     — IMAP + OAuth2 polling loop
│   └── outlook_producer.py   — Microsoft Graph API polling loop
├── admin/create_topics.py    — idempotent topic creation script
├── tests/test_producers.py   — 15 unit tests (classifier + schemas)
├── docker/docker-compose.yml — Kafka + ZooKeeper + Kafka UI
└── .env.example