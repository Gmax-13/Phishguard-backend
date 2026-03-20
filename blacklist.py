"""
blacklist.py — Domain & URL Blacklist Checker
==============================================

Two independent checks run in parallel:

1. SENDER DOMAIN CHECK (PhishTank)
   Extract the domain from the sender email and check it against
   PhishTank's verified phishing URL database. PhishTank is URL-based
   so we match the sender domain against any PhishTank entry that
   contains that domain.

2. EMAIL LINK CHECK (URLhaus + PhishTank)
   Each URL extracted from the email body is checked against:
     - URLhaus (abuse.ch) — malware/phishing URL feed
     - PhishTank — verified phishing URL database

Both feeds are cached locally in memory and refreshed every 6 hours
so we don't hit the external APIs on every single email scan.

If ANY check fires, the email is immediately flagged RED without
waiting for the HF Space inference — saving latency and API calls.
"""

import re
import csv
import logging
import requests
import threading
from io import StringIO
from datetime import datetime, timedelta
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Feed URLs
# -------------------------------------------------------

# URLhaus — CSV of active malicious URLs (no auth needed)
URLHAUS_CSV_URL = "https://urlhaus.abuse.ch/downloads/csv_recent/"

# PhishTank — JSON feed of verified phishing URLs (no auth for basic feed)
PHISHTANK_JSON_URL = "https://data.phishtank.com/data/online-valid.json"

# Cache refresh interval
CACHE_TTL_HOURS = 6

# -------------------------------------------------------
# In-memory cache
# -------------------------------------------------------

_cache = {
    "urlhaus_domains": set(),
    "urlhaus_urls":    set(),
    "phishtank_domains": set(),
    "phishtank_urls":    set(),
    "last_updated":    None,
    "lock":            threading.Lock(),
}


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _extract_domain(url_or_email: str) -> str:
    """
    Extract bare domain from a URL or email address.
    'http://evil.com/path' -> 'evil.com'
    'user@evil.com'        -> 'evil.com'
    'evil.com'             -> 'evil.com'
    """
    s = url_or_email.strip().lower()

    # Email address
    if "@" in s and "/" not in s:
        return s.split("@")[-1]

    # URL — add scheme if missing so urlparse works
    if not s.startswith("http"):
        s = "http://" + s

    parsed = urlparse(s)
    domain = parsed.netloc

    # Strip port
    domain = domain.split(":")[0]

    # Strip www.
    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def _normalise_url(url: str) -> str:
    """Lowercase and strip trailing slashes for consistent matching."""
    return url.strip().lower().rstrip("/")


# -------------------------------------------------------
# Feed loaders
# -------------------------------------------------------

def _load_urlhaus():
    """
    Fetch URLhaus recent CSV and extract domains + full URLs.
    CSV columns: id, dateadded, url, url_status, last_online, threat, tags, urlhaus_link, reporter
    Lines starting with # are comments.
    """
    domains = set()
    urls    = set()

    try:
        r = requests.get(URLHAUS_CSV_URL, timeout=20)
        r.raise_for_status()

        reader = csv.reader(StringIO(r.text))
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue

            raw_url = row[2].strip().strip('"')
            if not raw_url or not raw_url.startswith("http"):
                continue

            urls.add(_normalise_url(raw_url))
            domain = _extract_domain(raw_url)
            if domain:
                domains.add(domain)

        logger.info(f"URLhaus loaded: {len(urls)} URLs, {len(domains)} domains")

    except Exception as e:
        logger.warning(f"URLhaus feed load failed: {e}")

    return domains, urls


def _load_phishtank():
    """
    Fetch PhishTank JSON feed and extract domains + full URLs.
    Each entry: {"url": "...", "verified": "yes", ...}
    Note: The full feed does not require an API key.
    """
    domains = set()
    urls    = set()

    try:
        r = requests.get(
            PHISHTANK_JSON_URL,
            timeout=30,
            headers={"User-Agent": "PhishGuard/1.0 phishguard-project"}
        )
        r.raise_for_status()

        entries = r.json()

        for entry in entries:
            raw_url = entry.get("url", "").strip()
            if not raw_url:
                continue

            urls.add(_normalise_url(raw_url))
            domain = _extract_domain(raw_url)
            if domain:
                domains.add(domain)

        logger.info(f"PhishTank loaded: {len(urls)} URLs, {len(domains)} domains")

    except Exception as e:
        logger.warning(f"PhishTank feed load failed: {e}")

    return domains, urls


# -------------------------------------------------------
# Cache refresh
# -------------------------------------------------------

def _refresh_cache(force: bool = False):
    """
    Reload both feeds if cache is stale or forced.
    Thread-safe — uses a lock so concurrent requests don't
    trigger multiple simultaneous refreshes.
    """
    with _cache["lock"]:
        now = datetime.utcnow()

        if not force and _cache["last_updated"] is not None:
            age = now - _cache["last_updated"]
            if age < timedelta(hours=CACHE_TTL_HOURS):
                return  # Cache is fresh

        logger.info("Refreshing blacklist cache...")

        uh_domains, uh_urls       = _load_urlhaus()
        pt_domains, pt_urls       = _load_phishtank()

        _cache["urlhaus_domains"]   = uh_domains
        _cache["urlhaus_urls"]      = uh_urls
        _cache["phishtank_domains"] = pt_domains
        _cache["phishtank_urls"]    = pt_urls
        _cache["last_updated"]      = now

        total_domains = len(uh_domains | pt_domains)
        total_urls    = len(uh_urls | pt_urls)
        logger.info(f"Blacklist cache refreshed: {total_domains} domains, {total_urls} URLs")


def _ensure_cache():
    """Refresh cache if not yet loaded."""
    if _cache["last_updated"] is None:
        _refresh_cache(force=True)


# -------------------------------------------------------
# Check functions
# -------------------------------------------------------

def check_sender_domain(sender_email: str) -> dict:
    """
    Check if the sender's domain appears in either blacklist.

    Returns
    -------
    dict:
        {
          "flagged": bool,
          "domain":  str,
          "source":  "urlhaus" | "phishtank" | None
        }
    """
    _ensure_cache()

    domain = _extract_domain(sender_email)

    if not domain:
        return {"flagged": False, "domain": domain, "source": None}

    if domain in _cache["urlhaus_domains"]:
        return {"flagged": True, "domain": domain, "source": "URLhaus"}

    if domain in _cache["phishtank_domains"]:
        return {"flagged": True, "domain": domain, "source": "PhishTank"}

    return {"flagged": False, "domain": domain, "source": None}


def check_urls(links: list) -> dict:
    """
    Check each URL extracted from the email body against both blacklists.

    Parameters
    ----------
    links : list of dicts with "url" key (from email_preprocessing)

    Returns
    -------
    dict:
        {
          "flagged":       bool,
          "flagged_urls":  list of {"url": str, "domain": str, "source": str},
          "total_checked": int
        }
    """
    _ensure_cache()

    flagged_urls = []

    all_domains = _cache["urlhaus_domains"] | _cache["phishtank_domains"]
    all_urls    = _cache["urlhaus_urls"]    | _cache["phishtank_urls"]

    for link in links:
        raw_url = link.get("url", "").strip()
        if not raw_url:
            continue

        norm_url = _normalise_url(raw_url)
        domain   = _extract_domain(raw_url)

        # Full URL match
        if norm_url in all_urls:
            source = "URLhaus" if norm_url in _cache["urlhaus_urls"] else "PhishTank"
            flagged_urls.append({"url": raw_url, "domain": domain, "source": source})
            continue

        # Domain-only match (catches URL variations on a known bad domain)
        if domain and domain in all_domains:
            source = "URLhaus" if domain in _cache["urlhaus_domains"] else "PhishTank"
            flagged_urls.append({"url": raw_url, "domain": domain, "source": source})

    return {
        "flagged":       len(flagged_urls) > 0,
        "flagged_urls":  flagged_urls,
        "total_checked": len(links),
    }


# -------------------------------------------------------
# Combined check (single call from app.py)
# -------------------------------------------------------

def run_blacklist_checks(sender: str, links: list) -> dict:
    """
    Run both sender domain and URL checks.
    Returns a unified result dict.

    Parameters
    ----------
    sender : str   — raw sender email, e.g. "noreply@evil.com"
    links  : list  — list of {"url": ..., "text": ...} dicts

    Returns
    -------
    dict:
        {
          "blacklisted":     bool,   # True if ANY check fires
          "sender_check":    dict,   # result of check_sender_domain()
          "url_check":       dict,   # result of check_urls()
          "explanation":     str,    # human-readable summary
        }
    """
    sender_result = check_sender_domain(sender)
    url_result    = check_urls(links)

    blacklisted = sender_result["flagged"] or url_result["flagged"]

    # Build explanation string
    parts = []
    if sender_result["flagged"]:
        parts.append(
            f"Sender domain '{sender_result['domain']}' is on the "
            f"{sender_result['source']} blacklist."
        )
    if url_result["flagged"]:
        flagged_domains = list({f['domain'] for f in url_result["flagged_urls"]})
        sources         = list({f['source'] for f in url_result["flagged_urls"]})
        parts.append(
            f"{len(url_result['flagged_urls'])} link(s) match known phishing domains "
            f"({', '.join(flagged_domains)}) via {', '.join(sources)}."
        )

    explanation = " ".join(parts) if parts else "No blacklist matches found."

    return {
        "blacklisted": blacklisted,
        "sender_check": sender_result,
        "url_check":    url_result,
        "explanation":  explanation,
    }


# -------------------------------------------------------
# Background refresh thread
# Starts on import — refreshes every 6 hours automatically
# -------------------------------------------------------

def _background_refresh():
    """Run in a daemon thread to keep the cache warm."""
    import time
    while True:
        try:
            _refresh_cache()
        except Exception as e:
            logger.error(f"Background blacklist refresh error: {e}")
        # Sleep for slightly less than TTL so we always have fresh data
        time.sleep((CACHE_TTL_HOURS - 0.5) * 3600)


_refresh_thread = threading.Thread(target=_background_refresh, daemon=True)
_refresh_thread.start()