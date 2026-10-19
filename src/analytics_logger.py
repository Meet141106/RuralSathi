import json
import os
from datetime import datetime
from collections import Counter

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "analytics_log.json"
)


def load_log() -> list:
    """Load JSON array from LOG_PATH or return empty list on failure."""
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def log_event(event_type: str, data: dict) -> None:
    """Append event to log and cap at the last 500 events."""
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        log = load_log()
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            **data
        }
        log.append(event_record)
        if len(log) > 500:
            log = log[-500:]
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    except Exception:
        # Never crash application due to analytics logging failure
        pass


def detect_category(question: str) -> str:
    """Categorize user questions based on agronomic keywords."""
    if not question:
        return "General"
    q_lower = question.lower()

    if any(w in q_lower for w in ["fertilizer", "urea", "nitrogen", "phosphorus", "potash", "manure"]):
        return "Fertilizers"
    elif any(w in q_lower for w in ["pest", "insect", "aphid", "worm", "spray", "whitefly", "rust", "blast", "blight", "fungal", "disease"]):
        return "Pest Control"
    elif any(w in q_lower for w in ["scheme", "kisan", "pmfby", "subsidy", "government", "pm-kisan", "insurance", "loan", "kcc", "card"]):
        return "Government Schemes"
    elif any(w in q_lower for w in ["weather", "rain", "temperature", "irrigation", "water", "drought"]):
        return "Weather & Water"
    elif any(w in q_lower for w in ["wheat", "rice", "cotton", "sugarcane", "crop", "soil", "sow", "harvest", "seed", "paddy"]):
        return "Crop Management"
    return "General"


def get_summary() -> dict:
    """Aggregate statistics and metrics from the analytics log."""
    log = load_log()

    questions = [e for e in log if e.get("event") == "question_asked"]
    feedbacks = [e for e in log if e.get("event") == "feedback_given"]

    total_q = len(questions)
    abstentions = sum(1 for q in questions if q.get("abstained") is True)
    answered = total_q - abstentions
    abstention_rate = round((abstentions / total_q) * 100, 1) if total_q > 0 else 0.0

    pos_fb = sum(1 for f in feedbacks if f.get("positive") is True)
    neg_fb = sum(1 for f in feedbacks if f.get("positive") is False)

    # Categories breakdown
    categories = [q.get("category", "General") for q in questions]
    top_categories = dict(Counter(categories))

    # Questions by Hour (0-23)
    hour_counts = {h: 0 for h in range(24)}
    for q in questions:
        ts = q.get("timestamp")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                hour_counts[dt.hour] = hour_counts.get(dt.hour, 0) + 1
            except Exception:
                pass

    # Unanswered questions (last 10 abstained)
    unanswered = [
        q.get("question", "")
        for q in questions
        if q.get("abstained") is True and q.get("question")
    ][-10:]

    # Most asked topics (common words from questions excluding stopwords)
    stopwords = {"what", "how", "for", "the", "and", "in", "to", "is", "of", "a", "can", "i", "do", "should", "are", "about", "best", "use", "my"}
    all_words = []
    for q in questions:
        words = q.get("question", "").lower().replace("?", "").replace(",", "").split()
        all_words.extend([w for w in words if len(w) > 3 and w not in stopwords])

    most_asked = [w.capitalize() for w, _ in Counter(all_words).most_common(5)]
    if not most_asked:
        most_asked = ["Wheat", "Fertilizer", "Pest", "Weather", "Schemes"]

    return {
        "total_questions": total_q,
        "abstentions": abstentions,
        "answered": answered,
        "abstention_rate": abstention_rate,
        "positive_feedback": pos_fb,
        "negative_feedback": neg_fb,
        "top_categories": top_categories,
        "questions_by_hour": hour_counts,
        "unanswered_questions": unanswered,
        "most_asked_topics": most_asked
    }
