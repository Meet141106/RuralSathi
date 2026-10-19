import json
import os
import hashlib
from datetime import datetime

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "offline_cache.json"
)
MAX_CACHE = 20


def _hash_query(query: str) -> str:
    """Generate deterministic MD5 hash key for query."""
    return hashlib.md5(query.lower().strip().encode("utf-8")).hexdigest()[:12]


def load_cache() -> dict:
    """Load JSON dict of cached queries or return empty dict."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_cached(query: str) -> dict | None:
    """Retrieve cached answer by query string if present."""
    cache = load_cache()
    key = _hash_query(query)
    return cache.get(key, None)


def save_to_cache(query: str, answer: str, sources: list, conflict_result: dict = None) -> None:
    """Save Q&A pair to local offline cache, retaining the 20 most recent entries."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        cache = load_cache()
        key = _hash_query(query)
        cache[key] = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "conflict_result": conflict_result or {},
            "timestamp": datetime.now().isoformat(),
            "cached": True
        }
        if len(cache) > MAX_CACHE:
            oldest_key = sorted(cache, key=lambda k: cache[k].get("timestamp", ""))[0]
            del cache[oldest_key]

        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def get_all_cached() -> list:
    """Return all cached items sorted in reverse chronological order."""
    cache = load_cache()
    return sorted(cache.values(), key=lambda x: x.get("timestamp", ""), reverse=True)


def clear_cache() -> bool:
    """Delete the offline cache file."""
    try:
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        return True
    except Exception:
        return False


def is_online() -> bool:
    """Robust internet connectivity check with multi-endpoint fallback."""
    import socket
    import urllib.request

    # 1. Fast DNS socket probes (Cloudflare 1.1.1.1, Google 8.8.8.8)
    for host in [("1.1.1.1", 53), ("8.8.8.8", 53)]:
        try:
            sock = socket.create_connection(host, timeout=1.0)
            sock.close()
            return True
        except OSError:
            pass

    # 2. HTTP probe fallback (handles Wi-Fi/ISPs that block outbound UDP/TCP 53)
    for url in ["https://www.google.com", "https://1.1.1.1"]:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "RuralSathi/1.0"},
                method="HEAD"
            )
            with urllib.request.urlopen(req, timeout=1.5):
                return True
        except Exception:
            pass

    return False
