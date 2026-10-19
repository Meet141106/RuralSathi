import json
import os
from datetime import datetime

PROFILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "user_profile.json"
)

DEFAULT_PROFILE = {
    "name": "",
    "district": "Mumbai",
    "state": "Maharashtra",
    "crops": [],
    "language": "en",
    "land_size": "< 1 hectare",
    "has_aadhaar_bank": "Yes",
    "saved_questions": [],
    "feedback_given": 0
}


def load_profile() -> dict:
    """Load profile from PROFILE_PATH or return a copy of DEFAULT_PROFILE."""
    if not os.path.exists(PROFILE_PATH):
        return dict(DEFAULT_PROFILE)

    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged = dict(DEFAULT_PROFILE)
            merged.update(data)
            return merged
    except Exception:
        return dict(DEFAULT_PROFILE)


def save_profile(profile: dict) -> bool:
    """Write profile dict to data/user_profile.json."""
    try:
        os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def add_saved_question(question: str, answer: str) -> bool:
    """Append a Q&A record to saved_questions (capped at last 10)."""
    try:
        prof = load_profile()
        saved = prof.get("saved_questions", [])
        saved.append({
            "question": question,
            "answer": answer[:200],
            "timestamp": datetime.now().isoformat()
        })
        # Keep only the last 10
        prof["saved_questions"] = saved[-10:]
        return save_profile(prof)
    except Exception:
        return False


def get_personalized_greeting(profile: dict) -> str:
    """Return personalized greeting string based on profile."""
    name = (profile.get("name") or "").strip()
    if name:
        return f"Welcome back, {name}! 🌾"
    return "Welcome to RuralSathi 🌾"
