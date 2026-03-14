import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import html


# -----------------------------
# Helper: clean visible text
# -----------------------------
def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# Helper: extract visible text from HTML
# -----------------------------
def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return clean_text(text)


# -----------------------------
# Helper: extract URLs
# -----------------------------
def extract_urls(links):
    urls = []
    for link in links:
        url = link.get("url", "")
        if url:
            urls.append(url.strip())
    return urls


# -----------------------------
# URL feature extraction
# -----------------------------
def extract_url_features(urls):
    features = {
        "url_count": len(urls),
        "has_ip_url": 0,
        "has_punycode": 0,
        "long_url": 0,
        "suspicious_tld": 0
    }

    suspicious_tlds = {"zip", "mov", "click", "link", "xyz", "top"}

    for url in urls:
        parsed = urlparse(url)

        if re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc):
            features["has_ip_url"] = 1

        if "xn--" in parsed.netloc:
            features["has_punycode"] = 1

        if len(url) > 75:
            features["long_url"] = 1

        tld = parsed.netloc.split(".")[-1]
        if tld in suspicious_tlds:
            features["suspicious_tld"] = 1

    return features


# -----------------------------
# Image feature extraction (simple heuristic)
# -----------------------------
def extract_image_features(html_content, text_length):
    soup = BeautifulSoup(html_content, "html.parser")
    images = soup.find_all("img")

    return {
        "image_count": len(images),
        "text_length": text_length,
        "image_heavy": 1 if len(images) > 5 and text_length < 1000 else 0
    }


# -----------------------------
# 🚨 MAIN ENTRY POINT 🚨
# -----------------------------
def process_email_json(email_json: dict):
    """
    Converts extension JSON → plain text for ML/DL models
    """

    sender = email_json.get("sender", "")
    subject = email_json.get("subject", "")

    body = email_json.get("body", {})
    html_body = body.get("html", "")
    text_body = body.get("text", "")

    links = email_json.get("links", [])

    # 1️⃣ Extract visible text
    if html_body:
        visible_text = extract_text_from_html(html_body)
    else:
        visible_text = clean_text(text_body)

    # 2️⃣ Extract URLs
    urls = extract_urls(links)

    # 3️⃣ Compose final model text
    combined_text = f"""
Sender: {sender}
Subject: {subject}

{visible_text}

Links:
{" ".join(urls)}
""".strip()

    # 4️⃣ Feature extraction
    url_features = extract_url_features(urls)
    image_features = extract_image_features(html_body, len(visible_text))

    return {
    "sender": sender,
    "subject": subject,
    "clean_text": combined_text,
    "urls": urls,
    "url_features": url_features,
    "image_features": image_features
}