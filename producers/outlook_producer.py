from __future__ import annotations
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
 
import requests
 
from config.topics import (
    TOPIC_REJECTIONS,
    TOPIC_APPLICATIONS,
    TOPIC_DLQ_REJECTIONS,
    PARTITION_KEY_OUTLOOK,
)
from producers.base_producer import BaseProducer
from producers.classifier import classify_email, EmailCategory
from schemas.events import (
    EmailSource,
    make_rejection_event,
    make_application_event,
)
 
logger = logging.getLogger(__name__)
 
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TOKEN_URL  = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
DEVICE_URL = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode"
SCOPE      = "https://graph.microsoft.com/Mail.Read offline_access"
DEFAULT_POLL_INTERVAL = 60
 
 
# ── Token management ──────────────────────────────────────────────────────────
 
class _TokenManager:
    """Simple file-backed token cache with automatic refresh."""
 
    def __init__(self, client_id: str, tenant_id: str, token_path: str):
        self.client_id = client_id
        self.tenant_id = tenant_id
        self.token_path = Path(token_path)
        self._token_data: dict = {}
        self._load()
 
    def _load(self) -> None:
        if self.token_path.exists():
            self._token_data = json.loads(self.token_path.read_text())
 
    def _save(self) -> None:
        self.token_path.write_text(json.dumps(self._token_data, indent=2))
 
    def _is_expired(self) -> bool:
        exp = self._token_data.get("expires_at", 0)
        return time.time() >= exp - 60   # refresh 60s before real expiry
 
    def _refresh(self) -> None:
        url = TOKEN_URL.format(tenant=self.tenant_id)
        resp = requests.post(
            url,
            data={
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": self._token_data["refresh_token"],
                "scope": SCOPE,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        data["expires_at"] = time.time() + data.get("expires_in", 3600)
        self._token_data.update(data)
        self._save()
        logger.info("Outlook token refreshed")
 
    def device_code_flow(self) -> None:
        """One-time interactive authentication via device code."""
        device_url = DEVICE_URL.format(tenant=self.tenant_id)
        resp = requests.post(
            device_url,
            data={"client_id": self.client_id, "scope": SCOPE},
            timeout=15,
        )
        resp.raise_for_status()
        dc = resp.json()
        print(f"\n👉  Go to {dc['verification_uri']} and enter code: {dc['user_code']}\n")
 
        token_url = TOKEN_URL.format(tenant=self.tenant_id)
        interval = dc.get("interval", 5)
        while True:
            time.sleep(interval)
            poll = requests.post(
                token_url,
                data={
                    "client_id": self.client_id,
                    "grant_type": "urn:ietf:params:oauth2:grant-type:device_code",
                    "device_code": dc["device_code"],
                },
                timeout=15,
            )
            data = poll.json()
            if "access_token" in data:
                data["expires_at"] = time.time() + data.get("expires_in", 3600)
                self._token_data = data
                self._save()
                logger.info("Outlook device code flow complete, token saved")
                return
            if data.get("error") not in ("authorization_pending", "slow_down"):
                raise RuntimeError(f"Device code flow failed: {data}")
 
    def get_access_token(self) -> str:
        if not self._token_data:
            raise RuntimeError("No token found. Run with --auth first.")
        if self._is_expired():
            self._refresh()
        return self._token_data["access_token"]
 
 
# ── Outlook producer ──────────────────────────────────────────────────────────
 
class OutlookProducer(BaseProducer):
    """
    Polls Outlook (via Graph API) for unread job-related emails and
    publishes them to Kafka.
    """
 
    def __init__(
        self,
        kafka_config: dict,
        client_id: str,
        tenant_id: str,
        token_path: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        folder: str = "Inbox",
    ):
        super().__init__(
            kafka_config=kafka_config,
            producer_name="outlook_producer",
            dlq_topic=TOPIC_DLQ_REJECTIONS,
        )
        self._tokens = _TokenManager(client_id, tenant_id, token_path)
        self.poll_interval = poll_interval
        self.folder = folder
 
    # ── Graph API helpers ─────────────────────────────────────────────────────
 
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self._tokens.get_access_token()}",
            "Content-Type": "application/json",
        }
        resp = requests.get(
            f"{GRAPH_BASE}{path}",
            headers=headers,
            params=params,
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
 
    def _mark_read(self, message_id: str) -> None:
        headers = {
            "Authorization": f"Bearer {self._tokens.get_access_token()}",
            "Content-Type": "application/json",
        }
        requests.patch(
            f"{GRAPH_BASE}/me/messages/{message_id}",
            headers=headers,
            json={"isRead": True},
            timeout=10,
        )
 
    # ── Fetch and publish ─────────────────────────────────────────────────────
 
    def _fetch_unread(self) -> list[dict]:
        """
        Fetch unread messages from the configured folder.
        Uses $filter to only pull unread; $select to limit payload size.
        """
        data = self._get(
            "/me/mailFolders/Inbox/messages",
            params={
                "$filter": "isRead eq false",
                "$select": "id,subject,from,receivedDateTime,body",
                "$top": "50",
                "$orderby": "receivedDateTime asc",
            },
        )
        return data.get("value", [])
 
    def _process_message(self, msg: dict) -> None:
        subject     = msg.get("subject", "")
        sender      = msg.get("from", {}).get("emailAddress", {}).get("address", "")
        received_at = msg.get("receivedDateTime", "")
        body_data   = msg.get("body", {})
        body_text   = body_data.get("content", "")
        msg_id      = msg["id"]
 
        # Graph API returns HTML body by default; we requested text via $select
        # If content type is HTML, strip tags
        if body_data.get("contentType", "").lower() == "html":
            from html.parser import HTMLParser
            class S(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.parts = []
                def handle_data(self, d):
                    self.parts.append(d)
            p = S(); p.feed(body_text); body_text = " ".join(p.parts)
 
        category = classify_email(subject=subject, body=body_text)
 
        logger.debug(
            "Outlook | id=%s subject=%r category=%s", msg_id[:8], subject[:60], category
        )
 
        if category == EmailCategory.REJECTION:
            event = make_rejection_event(
                source=EmailSource.OUTLOOK,
                subject=subject,
                body_text=body_text,
                sender=sender,
                received_at=received_at,
            )
            self.send(
                topic=TOPIC_REJECTIONS,
                key=PARTITION_KEY_OUTLOOK,
                value=event.to_json(),
                headers={"schema_version": "1.0", "source": "outlook"},
            )
 
        elif category == EmailCategory.APPLICATION:
            event = make_application_event(
                source=EmailSource.OUTLOOK,
                subject=subject,
                body_text=body_text,
                sender=sender,
                received_at=received_at,
                job_board=self._infer_job_board(sender),
            )
            self.send(
                topic=TOPIC_APPLICATIONS,
                key=PARTITION_KEY_OUTLOOK,
                value=event.to_json(),
                headers={"schema_version": "1.0", "source": "outlook"},
            )
 
        self._mark_read(msg_id)
 
    @staticmethod
    def _infer_job_board(sender_email: str) -> str:
        sender = sender_email.lower()
        if "linkedin" in sender:    return "LinkedIn"
        if "indeed"   in sender:    return "Indeed"
        if "glassdoor" in sender:   return "Glassdoor"
        if "dice"     in sender:    return "Dice"
        if "ziprecruiter" in sender: return "ZipRecruiter"
        if "lever"    in sender:    return "Lever"
        if "greenhouse" in sender:  return "Greenhouse"
        if "workday"  in sender:    return "Workday"
        return "company_direct"
 
    # ── Poll loop ─────────────────────────────────────────────────────────────
 
    def run(self) -> None:
        logger.info(
            "Outlook producer started | poll_interval=%ds", self.poll_interval
        )
        try:
            while True:
                try:
                    messages = self._fetch_unread()
                    if messages:
                        logger.info(
                            "Outlook | %d unread message(s) to process", len(messages)
                        )
                    for msg in messages:
                        try:
                            self._process_message(msg)
                        except Exception as exc:
                            logger.error(
                                "Failed to process message id=%s: %s",
                                msg.get("id", "?")[:8], exc,
                            )
                except requests.RequestException as exc:
                    logger.error("Graph API error: %s — retrying after backoff", exc)
                    time.sleep(30)
                    continue
 
                time.sleep(self.poll_interval)
 
        except KeyboardInterrupt:
            logger.info("Outlook producer shutting down...")
        finally:
            self.flush()