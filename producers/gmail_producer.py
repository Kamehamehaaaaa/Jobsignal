from __future__ import annotations
import email
import imaplib
import logging
import os
import time
from email.header import decode_header
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Optional
 
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
 
from config.topics import (
    TOPIC_REJECTIONS,
    TOPIC_APPLICATIONS,
    TOPIC_DLQ_REJECTIONS,
    PARTITION_KEY_GMAIL,
)
from producers.base_producer import BaseProducer
from producers.classifier import classify_email, EmailCategory
from schemas.events import (
    EmailSource,
    make_rejection_event,
    make_application_event,
)


logger = logging.getLogger(__name__)
 
# Gmail IMAP endpoint
GMAIL_IMAP_HOST = "imap.gmail.com"
GMAIL_IMAP_PORT = 993
 
GMAIL_SCOPES = ["https://mail.google.com/"]
 
DEFAULT_POLL_INTERVAL = 60

MAX_EMAILS = 1000
 

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
 
    def handle_data(self, data: str):
        self._parts.append(data)
 
    def get_text(self) -> str:
        return " ".join(self._parts)
 
 
def _strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html)
    return s.get_text()

def _load_credentials(credentials_path: str, token_path: str) -> Credentials:
    creds: Optional[Credentials] = None
 
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)
 
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            logger.info("Gmail OAuth2 token refreshed")
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)
            logger.info("Gmail OAuth2 flow completed")
 
        with open(token_path, "w") as f:
            f.write(creds.to_json())
 
    return creds
 
 
def _build_xoauth2_string(user_email: str, access_token: str) -> str:
    return f"user={user_email}\x01auth=Bearer {access_token}\x01\x01"


def _decode_header_value(raw: str) -> str:
    parts = decode_header(raw)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)
 
 
def _extract_body(msg: email.message.Message) -> str:
    """Walk the MIME tree; prefer plain text, fall back to HTML-stripped."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
 
    for part in msg.walk():
        ct = part.get_content_type()
        if part.get_content_disposition() == "attachment":
            continue
        charset = part.get_content_charset() or "utf-8"
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        text = payload.decode(charset, errors="replace")
        if ct == "text/plain":
            plain_parts.append(text)
        elif ct == "text/html":
            html_parts.append(_strip_html(text))
 
    return ("\n".join(plain_parts) or "\n".join(html_parts)).strip()
 

class GmailProducer(BaseProducer):
    def __init__(
        self,
        kafka_config: dict,
        user_email: str,
        credentials_path: str,
        token_path: str,
        mailbox: str = "INBOX",
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ):
        super().__init__(
            kafka_config=kafka_config,
            producer_name="gmail_producer",
            dlq_topic=TOPIC_DLQ_REJECTIONS,
        )
        self.user_email = user_email
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.mailbox = mailbox
        self.poll_interval = poll_interval
        self._imap: Optional[imaplib.IMAP4_SSL] = None

    def _connect(self):
        creds = _load_credentials(self.credentials_path, self.token_path)
        auth_string = _build_xoauth2_string(self.user_email, creds.token)

        self._imap = imaplib.IMAP4_SSL(GMAIL_IMAP_HOST, GMAIL_IMAP_PORT)
        self._imap.authenticate("XOAUTH2", lambda _: auth_string.encode())
        self._imap.select(self.mailbox)
        logger.info("Connected to Gmail IMAP | user=%s", self.user_email)

    def _reconnect_if_needed(self) -> None:
        try:
            self._imap.noop()
        except Exception:
            logger.warning("IMAP connection lost, reconnecting...")
            self._connect()

    def _fetch_unread(self) -> list[str]:
        from datetime import datetime, timedelta
        self._reconnect_if_needed()

        since_date = (datetime.now() - timedelta(days=10)).strftime("%d-%b-%Y")
        _, data = self._imap.uid("search", None, "UNSEEN", "SINCE", since_date)
        uids = data[0].split() if data[0] else []

        uids = uids[-self.MAX_EMAILS:]
        return [uid.decode() for uid in uids]
 
    def _process_uid(self, uid: str) -> None:
        _, msg_data = self._imap.uid("fetch", uid, "(RFC822)")
        if not msg_data or not msg_data[0]:
            return
 
        raw_bytes = msg_data[0][1]
        msg = email.message_from_bytes(raw_bytes)
 
        subject = _decode_header_value(msg.get("Subject", ""))
        sender  = msg.get("From", "")
        date_str = msg.get("Date", "")
 
        try:
            received_at = parsedate_to_datetime(date_str).isoformat()
        except Exception:
            received_at = date_str
 
        body = _extract_body(msg)
        category = classify_email(subject=subject, body=body)
 
        logger.debug(
            "Gmail | uid=%s subject=%r category=%s", uid, subject[:60], category
        )
 
        if category == EmailCategory.REJECTION:
            event = make_rejection_event(
                source=EmailSource.GMAIL,
                subject=subject,
                body_text=body,
                sender=sender,
                received_at=received_at,
            )
            self.send(
                topic=TOPIC_REJECTIONS,
                key=PARTITION_KEY_GMAIL,
                value=event.to_json(),
                headers={"schema_version": "1.0", "source": "gmail"},
            )
 
        elif category == EmailCategory.APPLICATION:
            event = make_application_event(
                source=EmailSource.GMAIL,
                subject=subject,
                body_text=body,
                sender=sender,
                received_at=received_at,
            )
            self.send(
                topic=TOPIC_APPLICATIONS,
                key=PARTITION_KEY_GMAIL,
                value=event.to_json(),
                headers={"schema_version": "1.0", "source": "gmail"},
            )
        else:
            logger.debug("Gmail | uid=%s — skipped (not job-related)", uid)
 
        # Mark as seen so we don't reprocess on next poll
        self._imap.uid("store", uid, "+FLAGS", "\\Seen")
 
    # ── Poll loop ─────────────────────────────────────────────────────────────
 
    def run(self) -> None:
        """
        Blocking poll loop.  Run this in a thread or as a standalone process.
        Ctrl-C (KeyboardInterrupt) triggers a clean flush and exit.
        """
        self._connect()
        logger.info(
            "Gmail producer started | poll_interval=%ds", self.poll_interval
        )
        try:
            while True:
                uids = self._fetch_unread()
                if uids:
                    logger.info("Gmail | %d unread message(s) to process", len(uids))
                for uid in uids:
                    try:
                        self._process_uid(uid)
                    except Exception as exc:
                        logger.error("Failed to process uid=%s: %s", uid, exc)
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Gmail producer shutting down...")
        finally:
            if self._imap:
                try:
                    self._imap.logout()
                except Exception:
                    pass
            self.flush()
 

if __name__ == "__main__":
    import logging
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(levelname)s %(name)s — %(message)s",
    )

    producer = GmailProducer(
        kafka_config={
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        },
        user_email=os.getenv("GMAIL_USER_EMAIL"),
        credentials_path=os.getenv("GMAIL_CREDENTIALS_PATH", "secrets/gmail_credentials.json"),
        token_path=os.getenv("GMAIL_TOKEN_PATH", "secrets/gmail_token.json"),
        poll_interval=int(os.getenv("GMAIL_POLL_INTERVAL", "60")),
    )
    producer.run()