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

# OpenPhish — plain text feed, one URL per line, no auth needed
# Updated every 12 hours, contains active phishing URLs
OPENPHISH_TXT_URL = "https://openphish.com/feed.txt"

# Cache refresh interval
CACHE_TTL_HOURS = 6

# -------------------------------------------------------
# In-memory cache
# -------------------------------------------------------

_cache = {
    "urlhaus_domains":   set(),
    "urlhaus_urls":      set(),
    "urlhaus_base_urls": set(),   # scheme+host+port only, for prefix matching
    "openphish_domains": set(),
    "openphish_urls":    set(),
    "openphish_base_urls": set(),
    "last_updated":      None,
    "lock":              threading.Lock(),
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


def _extract_base_url(url: str) -> str:
    """Extract scheme+host+port — used for prefix matching.
    'http://27.153.152.117:41069/path' -> 'http://27.153.152.117:41069'
    """
    try:
        parsed = urlparse(url.strip().lower())
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return ""


# -------------------------------------------------------
# Feed loaders
# -------------------------------------------------------

def _load_urlhaus():
    """
    Fetch URLhaus recent CSV and extract domains + full URLs.
    CSV columns: id, dateadded, url, url_status, last_online, threat, tags, urlhaus_link, reporter
    Lines starting with # are comments.
    """
    domains   = set()
    urls      = set()
    base_urls = set()

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
            base = _extract_base_url(raw_url)
            if base:
                base_urls.add(base)

        logger.info(f"URLhaus loaded: {len(urls)} URLs, {len(domains)} domains")

    except Exception as e:
        logger.warning(f"URLhaus feed load failed: {e}")

    return domains, urls, base_urls


def _load_openphish():
    domains   = set()
    urls      = set()
    base_urls = set()

    try:
        r = requests.get(
            OPENPHISH_TXT_URL,
            timeout=20,
            headers={"User-Agent": "PhishGuard/1.0 phishguard-project"}
        )
        r.raise_for_status()

        for line in r.text.splitlines():
            raw_url = line.strip()
            if not raw_url or not raw_url.startswith("http"):
                continue

            urls.add(_normalise_url(raw_url))
            domain = _extract_domain(raw_url)
            if domain:
                domains.add(domain)
            base = _extract_base_url(raw_url)
            if base:
                base_urls.add(base)

        logger.info(f"OpenPhish loaded: {len(urls)} URLs, {len(domains)} domains")

    except Exception as e:
        logger.warning(f"OpenPhish feed load failed: {e}")

    return domains, urls, base_urls


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

        uh_domains, uh_urls, uh_base       = _load_urlhaus()
        op_domains, op_urls, op_base       = _load_openphish()

        _cache["urlhaus_domains"]    = uh_domains
        _cache["urlhaus_urls"]       = uh_urls
        _cache["urlhaus_base_urls"]  = uh_base
        _cache["openphish_domains"]  = op_domains
        _cache["openphish_urls"]     = op_urls
        _cache["openphish_base_urls"]= op_base
        _cache["last_updated"]       = now

        total_domains = len(uh_domains | op_domains)
        total_urls    = len(uh_urls | op_urls)
        logger.info(f"Blacklist cache refreshed: {total_domains} domains, {total_urls} URLs")


def _ensure_cache() -> bool:
    """Return True if cache is ready. Triggers a background load if not yet started."""
    return _cache["last_updated"] is not None


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
    if not _ensure_cache():
        return {"flagged": False, "domain": _extract_domain(sender_email), "source": None}

    domain = _extract_domain(sender_email)

    if not domain:
        return {"flagged": False, "domain": domain, "source": None}

    if domain in _cache["urlhaus_domains"]:
        return {"flagged": True, "domain": domain, "source": "URLhaus"}

    if domain in _cache["openphish_domains"]:
        return {"flagged": True, "domain": domain, "source": "OpenPhish"}

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
    if not _ensure_cache():
        return {"flagged": False, "flagged_urls": [], "total_checked": len(links)}

    flagged_urls = []

    all_domains = _cache["urlhaus_domains"] | _cache["openphish_domains"]
    all_urls    = _cache["urlhaus_urls"]    | _cache["openphish_urls"]

    for link in links:
        raw_url = link.get("url", "").strip()
        if not raw_url:
            continue

        norm_url = _normalise_url(raw_url)
        domain   = _extract_domain(raw_url)
        flagged  = False
        source   = None

        # 1. Exact full URL match
        if norm_url in all_urls:
            source  = "URLhaus" if norm_url in _cache["urlhaus_urls"] else "OpenPhish"
            flagged = True

        # 2. Domain-only match (catches URL path variations on a known bad domain)
        elif domain and domain in all_domains:
            source  = "URLhaus" if domain in _cache["urlhaus_domains"] else "OpenPhish"
            flagged = True

        # 3. Base URL match (scheme+host+port) — catches path variations
        # e.g. URLhaus has "http://27.153.152.117:41069", email has "http://27.153.152.117:41069/i"
        else:
            base = _extract_base_url(raw_url)
            all_bases = _cache["urlhaus_base_urls"] | _cache["openphish_base_urls"]
            if base and base in all_bases:
                source  = "URLhaus" if base in _cache["urlhaus_base_urls"] else "OpenPhish"
                flagged = True

        if flagged:
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
# Delays initial load by 10 seconds so Render's health
# check passes before any network calls are made.
# All feed errors are caught and logged — never crashes.
# -------------------------------------------------------

def _background_refresh():
    """Periodically refresh the cache every 6 hours."""
    import time
    # First sleep is the full TTL since app.py already loaded on startup
    time.sleep(CACHE_TTL_HOURS * 3600)
    while True:
        try:
            _refresh_cache()
        except Exception as e:
            logger.error(f"Background blacklist refresh error: {e}")
        time.sleep(CACHE_TTL_HOURS * 3600)


_refresh_thread = threading.Thread(target=_background_refresh, daemon=True)
_refresh_thread.start()