"""Send a report by email over SMTP. Qt-free (stdlib ``smtplib`` + ``email``).

Used by the scheduled Morning Report delivery. The password is supplied by the
caller (read from the OS keychain), never stored here. Supports STARTTLS
(port 587, the common default) and implicit SSL (port 465).
"""

from __future__ import annotations

import mimetypes
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path


def _clean_header(value: str) -> str:
    """Strip CR/LF so a crafted subject/recipient can't inject SMTP headers."""
    return (value or "").replace("\r", " ").replace("\n", " ").strip()


@dataclass
class SmtpConfig:
    host: str
    port: int
    username: str
    password: str
    from_addr: str = ""
    use_ssl: bool = False  # True → SMTP_SSL (465); False → STARTTLS (587)
    timeout: int = 30

    @property
    def sender(self) -> str:
        return self.from_addr or self.username


def _build_message(cfg: SmtpConfig, to_addrs, subject: str, html_body: str,
                   attachments=None) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = _clean_header(subject)
    msg["From"] = _clean_header(cfg.sender)
    to_list = [to_addrs] if isinstance(to_addrs, str) else list(to_addrs)
    msg["To"] = ", ".join(_clean_header(a) for a in to_list if a and a.strip())
    # Plain-text fallback + the real HTML body.
    msg.set_content("Your Portfolio Analyzer morning brief is attached / below. "
                    "View this message in an HTML-capable client.")
    msg.add_alternative(html_body, subtype="html")
    for path in (attachments or []):
        p = Path(path)
        if not p.exists():
            continue
        ctype, _ = mimetypes.guess_type(p.name)
        maintype, _, subtype = (ctype or "application/octet-stream").partition("/")
        msg.add_attachment(p.read_bytes(), maintype=maintype,
                           subtype=subtype or "octet-stream", filename=p.name)
    return msg


def send_email(cfg: SmtpConfig, to_addrs, subject: str, html_body: str,
               attachments=None) -> None:
    """Send an HTML email (optionally with attachments). Raises on failure so the
    caller can surface the exact SMTP error."""
    msg = _build_message(cfg, to_addrs, subject, html_body, attachments)
    # Verify the server certificate + hostname (default context) so an on-path
    # attacker can't present a rogue cert and capture the SMTP credential.
    ctx = ssl.create_default_context()
    if cfg.use_ssl:
        with smtplib.SMTP_SSL(cfg.host, int(cfg.port), timeout=cfg.timeout, context=ctx) as s:
            s.login(cfg.username, cfg.password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(cfg.host, int(cfg.port), timeout=cfg.timeout) as s:
            s.ehlo()
            s.starttls(context=ctx)
            s.ehlo()
            s.login(cfg.username, cfg.password)
            s.send_message(msg)
