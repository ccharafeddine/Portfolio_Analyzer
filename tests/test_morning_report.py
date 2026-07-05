"""Morning report: brief builder, SMTP emailer, keychain helper (all Qt-free)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.reports import emailer
from src.reports.emailer import SmtpConfig, send_email
from src.reports.morning_brief import build_morning_brief, compute_brief


def _q(**kw):
    return SimpleNamespace(**kw)


# ── Morning Brief builder ──────────────────────────────────────────
def test_compute_brief_weighted_change_and_pnl_uses_invested():
    cfg = SimpleNamespace(
        tickers=["AAPL", "MSFT"], weights={"AAPL": 0.5, "MSFT": 0.5},
        capital=100_000.0, cash=20_000.0, cost_basis={},
    )
    quotes = {"AAPL": _q(change_pct=0.02, last=110.0),
              "MSFT": _q(change_pct=-0.01, last=95.0)}
    d = compute_brief(cfg, quotes, name="X")
    assert abs(d.day_pct - 0.005) < 1e-9          # 0.5*.02 + 0.5*(-.01)
    assert abs(d.day_pnl - 400.0) < 1e-6          # invested 80k * 0.005
    assert d.total_value == 100_000.0
    assert len(d.holdings) == 2


def test_compute_brief_handles_missing_quotes():
    cfg = SimpleNamespace(tickers=["AAPL"], weights={"AAPL": 1.0},
                          capital=1000.0, cash=0.0, cost_basis={})
    d = compute_brief(cfg, {}, name="X")   # no quotes at all
    assert d.day_pct is None and d.day_pnl is None
    assert d.holdings[0]["chg_pct"] == "—" and d.holdings[0]["neutral"] is True


def test_build_morning_brief_escapes_untrusted_and_structures():
    cfg = SimpleNamespace(tickers=["AAPL"], weights={"AAPL": 1.0},
                          capital=1000.0, cash=0.0, cost_basis={})
    news = [SimpleNamespace(title="<script>alert(1)</script>",
                            url="javascript:alert(1)", publisher="EvilWire", published=None)]
    out = build_morning_brief(cfg, {"AAPL": _q(change_pct=0.01, last=10.0)},
                              news=news, name="MyPort")
    html = out["html"]
    assert html.startswith("<!DOCTYPE html>") and html.rstrip().endswith("</html>")
    assert "<script>alert(1)</script>" not in html       # escaped by autoescape
    assert "&lt;script&gt;" in html
    assert "javascript:alert(1)" not in html             # non-http url dropped
    assert "Content-Security-Policy" in html
    assert out["subject"].startswith("Morning Brief · MyPort")
    assert "day change" in out["summary"]


# ── SMTP emailer ───────────────────────────────────────────────────
def _ctx(instance):
    cm = MagicMock()
    cm.__enter__.return_value = instance
    cm.__exit__.return_value = False
    return cm


def test_send_email_starttls_path():
    cfg = SmtpConfig(host="smtp.x", port=587, username="u@x", password="pw")
    inst = MagicMock()
    with patch.object(emailer.smtplib, "SMTP", return_value=_ctx(inst)) as mk:
        send_email(cfg, "to@x", "Subj", "<p>hi</p>")
    mk.assert_called_once()
    inst.starttls.assert_called_once()
    inst.login.assert_called_once_with("u@x", "pw")
    inst.send_message.assert_called_once()


def test_send_email_ssl_path_and_multiple_recipients():
    cfg = SmtpConfig(host="smtp.x", port=465, username="u@x", password="pw", use_ssl=True)
    inst = MagicMock()
    with patch.object(emailer.smtplib, "SMTP_SSL", return_value=_ctx(inst)) as mk:
        send_email(cfg, ["a@x", "b@x"], "S", "<p>x</p>")
    mk.assert_called_once()
    inst.login.assert_called_once()
    inst.send_message.assert_called_once()


def test_send_email_verifies_tls_context():
    """Both TLS paths must pass an SSL context so the server cert is verified."""
    inst = MagicMock()
    with patch.object(emailer.smtplib, "SMTP", return_value=_ctx(inst)):
        send_email(SmtpConfig(host="smtp.x", port=587, username="u@x", password="pw"),
                   "to@x", "Subj", "<p>hi</p>")
    assert inst.starttls.call_args.kwargs.get("context") is not None

    with patch.object(emailer.smtplib, "SMTP_SSL", return_value=_ctx(MagicMock())) as mk:
        send_email(SmtpConfig(host="smtp.x", port=465, username="u@x", password="pw",
                              use_ssl=True), "to@x", "S", "<p>x</p>")
    assert mk.call_args.kwargs.get("context") is not None


def test_build_message_strips_header_crlf_injection():
    """A CRLF in the subject/recipient must not smuggle extra SMTP headers."""
    cfg = SmtpConfig(host="h", port=587, username="u@x", password="pw")
    msg = emailer._build_message(
        cfg, "victim@x\r\nBcc: evil@x", "Hi\r\nX-Injected: 1", "<p>hi</p>")
    for h in ("Subject", "To", "From"):
        assert "\r" not in (msg[h] or "") and "\n" not in (msg[h] or "")
    assert msg.get_all("Bcc") is None
    assert msg.get_all("X-Injected") is None


def test_build_message_has_html_alt_and_pdf_attachment(tmp_path):
    p = tmp_path / "report.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    cfg = SmtpConfig(host="h", port=587, username="u@x", password="pw")
    msg = emailer._build_message(cfg, "to@x", "Subj", "<p>hi</p>", attachments=[str(p)])
    assert msg["Subject"] == "Subj" and msg["From"] == "u@x" and msg["To"] == "to@x"
    parts = list(msg.walk())
    assert any(m.get_content_type() == "text/html" for m in parts)
    assert any(m.get_filename() == "report.pdf" for m in parts)


# ── Keychain helper (skips where no OS backend, e.g. headless CI) ──
def test_keychain_password_roundtrip_if_available():
    from src.ui import settings as st

    if not st.keyring_available():
        pytest.skip("no OS keychain backend available")
    user = "pa-unit-test@example.invalid"
    assert st.set_email_password(user, "secret-xyz") is True
    assert st.get_email_password(user) == "secret-xyz"
    st.delete_email_password(user)
    assert st.get_email_password(user) is None
