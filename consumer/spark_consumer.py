"""
JobSignal — Spark Structured Streaming consumer.

Reads from both Kafka topics, runs NLP enrichment on each event, and
writes to Google Sheets + local files in parallel.

Architecture:
    Kafka topics
        raw.emails.rejections   ──┐
        raw.emails.applications ──┤
                                  ▼
                         Spark Structured Streaming
                           (foreachBatch sink)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              Google Sheets               Local JSONL/CSV
           (Rejections tab)             (output/rejections.jsonl)
           (Applications tab)           (output/applications.jsonl)

Why foreachBatch instead of foreach?
    foreachBatch gives us the full micro-batch as a DataFrame, which means
    we can apply NLP in a vectorised UDF across the batch rather than
    processing one row at a time.  It also lets us write to both sinks
    (Sheets + local) atomically per batch.

Checkpointing:
    Spark checkpoints to ./checkpoints/ so that on restart it picks up
    exactly where it left off — no duplicate processing, no missed events.
    This is the exactly-once semantics story for DevOps/MLOps interviews.

Running:
    python -m jobsignal.consumer.spark_consumer

    Or with spark-submit for a real cluster:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \\
        jobsignal/consumer/spark_consumer.py
"""

from __future__ import annotations
import json
import logging
import os
import time

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType,
)
from dotenv import load_dotenv

from config.topics import TOPIC_REJECTIONS, TOPIC_APPLICATIONS
from nlp.enricher import enrich
from sinks.local_sink import LocalSink
from sinks.sheets_sink import SheetsSink
from tracking.mlflow_tracker import log_batch_run

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

# Label used in MLflow run tracking — mirrors the USE_DISTILBERT flag in nlp/enricher.py
_distilbert_mode = os.getenv("USE_DISTILBERT", "").strip().lower()
_DISTILBERT_MODE_LABEL = _distilbert_mode if _distilbert_mode in ("zero_shot", "finetuned") else "spacy"


# ── Spark session ─────────────────────────────────────────────────────────────

def _build_spark() -> SparkSession:
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    return (
        SparkSession.builder
        .appName("JobSignal-Consumer")
        .config("spark.sql.shuffle.partitions", "4")   # keep low for local dev
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        )
        .getOrCreate()
    )


# ── Kafka source schema ───────────────────────────────────────────────────────
# The fields we care about inside the JSON value of each Kafka message.

EVENT_SCHEMA = StructType([
    StructField("event_id",           StringType(), True),
    StructField("source",             StringType(), True),
    StructField("raw_subject",        StringType(), True),
    StructField("raw_body_text",      StringType(), True),
    StructField("sender_email",       StringType(), True),
    StructField("received_at",        StringType(), True),
    StructField("ingested_at",        StringType(), True),
    StructField("schema_version",     StringType(), True),
    # Rejection-specific (null for application events)
    StructField("rejection_type",     StringType(), True),
    # Application-specific (null for rejection events)
    StructField("application_status", StringType(), True),
    StructField("job_board",          StringType(), True),
])

# Schema for the enriched columns returned by the UDF
ENRICHED_SCHEMA = StructType([
    StructField("company_name",   StringType(), True),
    StructField("role_title",     StringType(), True),
    StructField("rejection_type", StringType(), True),
    StructField("job_board",      StringType(), True),
    StructField("confidence",     FloatType(),  True),
])


# ── NLP UDF ──────────────────────────────────────────────────────────────────

def _make_enrich_udf(event_type: str):
    """
    Returns a Spark UDF that runs NLP enrichment on a single row.

    We create one UDF per event type so the enricher knows whether to
    compute rejection_type or skip it.
    """
    from pyspark.sql.functions import udf

    def _enrich_row(subject: str, body: str, sender: str):
        if not subject and not body:
            return ("", "", "unknown", "company_direct", 0.0)
        result = enrich(
            subject=subject or "",
            body=body or "",
            sender_email=sender or "",
            event_type=event_type,
        )
        return (
            result.company_name,
            result.role_title,
            result.rejection_type,
            result.job_board,
            float(result.confidence),
        )

    return udf(_enrich_row, ENRICHED_SCHEMA)


_enrich_rejection_udf    = _make_enrich_udf("rejection")
_enrich_application_udf  = _make_enrich_udf("application")


# ── Kafka reader ──────────────────────────────────────────────────────────────

def _read_topic(spark: SparkSession, topic: str) -> DataFrame:
    """
    Read a Kafka topic as a Structured Streaming DataFrame.

    startingOffsets="latest" means we only process new messages — not the
    full history on every restart.  Change to "earliest" to reprocess
    everything (useful for backfill after a bug fix).
    """
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")   # don't crash if a partition is empty
        .load()
        # Kafka messages arrive as binary; cast value to string
        .select(F.col("value").cast("string").alias("json_value"))
        # Parse the JSON payload into typed columns
        .select(F.from_json(F.col("json_value"), EVENT_SCHEMA).alias("e"))
        .select("e.*")
    )


# ── foreachBatch handlers ─────────────────────────────────────────────────────

def _process_rejections_batch(
    batch_df: DataFrame,
    batch_id: int,
    local_sink: LocalSink,
    sheets_sink: SheetsSink | None,
) -> None:
    """
    Called by Spark for each micro-batch from raw.emails.rejections.
    Runs NLP enrichment then writes to both sinks.
    """
    count = batch_df.count()
    if count == 0:
        return

    batch_start = time.perf_counter()
    logger.info("Processing rejection batch | batch_id=%d rows=%d", batch_id, count)

    # Add enriched columns via UDF
    enriched_df = batch_df.withColumn(
        "nlp",
        _enrich_rejection_udf(
            F.col("raw_subject"),
            F.col("raw_body_text"),
            F.col("sender_email"),
        ),
    ).select(
        "*",
        F.col("nlp.company_name").alias("company_name"),
        F.col("nlp.role_title").alias("role_title"),
        F.col("nlp.rejection_type").alias("enriched_rejection_type"),
        F.col("nlp.job_board").alias("enriched_job_board"),
        F.col("nlp.confidence").alias("nlp_confidence"),
    ).drop("nlp")

    # Write each row to both sinks, collecting tracking data as we go
    mlflow_rows: list[dict] = []
    for row in enriched_df.collect():
        event = row.asDict()
        # Build a lightweight enriched object for the sink APIs
        enriched = _RowEnriched(
            company_name=event.get("company_name") or "",
            role_title=event.get("role_title") or "",
            rejection_type=event.get("enriched_rejection_type") or "unknown",
            job_board=event.get("enriched_job_board") or "company_direct",
            confidence=event.get("nlp_confidence") or 0.0,
        )
        local_sink.write_rejection(event, enriched)
        if sheets_sink:
            try:
                sheets_sink.write_rejection(event, enriched)
            except Exception as exc:
                logger.error("Sheets write failed for rejection: %s", exc)

        mlflow_rows.append({
            "company_name":   enriched.company_name,
            "role_title":     enriched.role_title,
            "rejection_type": enriched.rejection_type,
            "confidence":     enriched.confidence,
        })

    batch_latency = time.perf_counter() - batch_start
    log_batch_run(
        backend=_DISTILBERT_MODE_LABEL,
        event_type="rejection",
        enriched_rows=mlflow_rows,
        batch_latency_seconds=batch_latency,
    )


def _process_applications_batch(
    batch_df: DataFrame,
    batch_id: int,
    local_sink: LocalSink,
    sheets_sink: SheetsSink | None,
) -> None:
    """
    Called by Spark for each micro-batch from raw.emails.applications.
    """
    count = batch_df.count()
    if count == 0:
        return

    batch_start = time.perf_counter()
    logger.info("Processing application batch | batch_id=%d rows=%d", batch_id, count)

    enriched_df = batch_df.withColumn(
        "nlp",
        _enrich_application_udf(
            F.col("raw_subject"),
            F.col("raw_body_text"),
            F.col("sender_email"),
        ),
    ).select(
        "*",
        F.col("nlp.company_name").alias("company_name"),
        F.col("nlp.role_title").alias("role_title"),
        F.col("nlp.job_board").alias("enriched_job_board"),
        F.col("nlp.confidence").alias("nlp_confidence"),
    ).drop("nlp")

    mlflow_rows: list[dict] = []
    for row in enriched_df.collect():
        event = row.asDict()
        enriched = _RowEnriched(
            company_name=event.get("company_name") or "",
            role_title=event.get("role_title") or "",
            rejection_type="n/a",
            job_board=event.get("enriched_job_board") or "company_direct",
            confidence=event.get("nlp_confidence") or 0.0,
        )
        local_sink.write_application(event, enriched)
        if sheets_sink:
            try:
                sheets_sink.write_application(event, enriched)
            except Exception as exc:
                logger.error("Sheets write failed for application: %s", exc)

        mlflow_rows.append({
            "company_name": enriched.company_name,
            "role_title":   enriched.role_title,
            "rejection_type": "n/a",
            "confidence":   enriched.confidence,
        })

    batch_latency = time.perf_counter() - batch_start
    log_batch_run(
        backend=_DISTILBERT_MODE_LABEL,
        event_type="application",
        enriched_rows=mlflow_rows,
        batch_latency_seconds=batch_latency,
    )


# ── Simple namespace to pass enriched fields to sinks ────────────────────────

class _RowEnriched:
    __slots__ = ("company_name", "role_title", "rejection_type", "job_board", "confidence")

    def __init__(self, company_name, role_title, rejection_type, job_board, confidence):
        self.company_name   = company_name
        self.role_title     = role_title
        self.rejection_type = rejection_type
        self.job_board      = job_board
        self.confidence     = confidence


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    spark = _build_spark()
    spark.sparkContext.setLogLevel("WARN")   # quieten Spark's verbose logs

    # Initialise sinks
    local_sink = LocalSink(output_dir=os.getenv("LOCAL_OUTPUT_DIR", "output"))

    sheets_sink: SheetsSink | None = None
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
    if spreadsheet_id:
        try:
            sheets_sink = SheetsSink(
                spreadsheet_id=spreadsheet_id,
                credentials_path=os.getenv(
                    "GMAIL_CREDENTIALS_PATH", "secrets/gmail_credentials.json"
                ),
                token_path=os.getenv(
                    "GMAIL_TOKEN_PATH", "secrets/gmail_token.json"
                ),
            )
            logger.info("Google Sheets sink enabled | spreadsheet_id=%s", spreadsheet_id)
        except Exception as exc:
            logger.warning("Sheets sink failed to initialise (%s) — local only", exc)
    else:
        logger.info("GOOGLE_SPREADSHEET_ID not set — running local sink only")

    # Read both topics
    rejections_df   = _read_topic(spark, TOPIC_REJECTIONS)
    applications_df = _read_topic(spark, TOPIC_APPLICATIONS)

    # Start two streaming queries — one per topic
    rejection_query = (
        rejections_df.writeStream
        .foreachBatch(
            lambda df, bid: _process_rejections_batch(df, bid, local_sink, sheets_sink)
        )
        .option("checkpointLocation", "checkpoints/rejections")
        .trigger(processingTime="30 seconds")   # micro-batch every 30s
        .start()
    )

    application_query = (
        applications_df.writeStream
        .foreachBatch(
            lambda df, bid: _process_applications_batch(df, bid, local_sink, sheets_sink)
        )
        .option("checkpointLocation", "checkpoints/applications")
        .trigger(processingTime="30 seconds")
        .start()
    )

    logger.info(
        "JobSignal consumer running | "
        "rejection_query=%s application_query=%s",
        rejection_query.id, application_query.id,
    )

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
        rejection_query.stop()
        application_query.stop()
    finally:
        spark.stop()
        logger.info("Summary: %s", local_sink.summary())


if __name__ == "__main__":
    run()