import logging
import urllib.parse
import urllib.request
import json
from deep_translator import GoogleTranslator
from langdetect import detect

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect if text is Hindi ('hi'), Marathi ('mr'), or default to English ('en')."""
    if not text or not text.strip():
        return "en"
    try:
        lang = detect(text.strip())
        if lang in ["hi", "mr"]:
            return lang
        return "en"
    except Exception as e:
        logger.debug("Language detection failed: %s", e)
        return "en"


def _fallback_translate(text: str, target_lang: str, source_lang: str = "auto") -> str:
    """High-reliability fallback using Google GTX API endpoint."""
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return "".join([part[0] for part in payload[0] if part[0]])
    except Exception as e:
        logger.warning("Fallback translate failed: %s", e)
        return text


def translate_to_english(text: str, source_lang: str = None) -> str:
    """Translate user query to English for vector retrieval."""
    if not text or not text.strip():
        return text

    if not source_lang or source_lang == "auto":
        source_lang = detect_language(text)

    if source_lang == "en":
        return text

    # Attempt 1: Deep Translator
    try:
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        if translated:
            return translated
    except Exception:
        pass

    # Attempt 2: Fallback GTX
    return _fallback_translate(text, target_lang="en", source_lang=source_lang)


def translate_answer(text: str, target_lang: str) -> str:
    """Translate answer text to target language (e.g. 'hi' or 'mr')."""
    if not text or not text.strip() or target_lang == "en":
        return text

    # Attempt 1: Deep Translator
    try:
        if len(text) < 4000:
            translated = GoogleTranslator(source="en", target=target_lang).translate(text)
            if translated:
                return translated
    except Exception:
        pass

    # Attempt 2: Fallback GTX
    res = _fallback_translate(text, target_lang=target_lang, source_lang="en")
    if res and res != text:
        return res

    return f"{text}\n\n*(Translation to {target_lang} unavailable)*"
