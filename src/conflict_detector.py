import re
from difflib import SequenceMatcher

CONFLICT_KEYWORDS = {
    "dose": ["kg/ha", "kg per acre", "ml/litre", "grams", "kg/hectare", "ml/acre", "g/litre", "g/acre"],
    "timing": ["days after sowing", "das", "days after germination", "weeks after", "stage"],
    "eligibility": ["hectares", "acres", "income", "annual", "per year", "rs", "₹", "rupees"],
}


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from a given text snippet."""
    if not text:
        return []
    matches = re.findall(r'\b\d+\.?\d*\b', text)
    nums = []
    for m in matches:
        try:
            val = float(m)
            # Filter out non-actionable extremes
            if 0.01 <= val <= 100000:
                nums.append(val)
        except ValueError:
            continue
    return nums


def check_numeric_conflict(chunks: list, tolerance: float = 0.3) -> dict:
    """Detect if multiple chunks present values for the same parameter differing by > tolerance (30%)."""
    conflicts = []

    for cat, keywords in CONFLICT_KEYWORDS.items():
        matched_chunks = []
        for c in chunks:
            text_lower = c.get("text", "").lower()
            if any(kw in text_lower for kw in keywords):
                nums = extract_numbers(c.get("text", ""))
                if nums:
                    matched_chunks.append({
                        "source": c.get("source", "Document"),
                        "values": nums
                    })

        if len(matched_chunks) > 1:
            all_vals = []
            for mc in matched_chunks:
                all_vals.extend(mc["values"])

            if all_vals:
                min_val = min(all_vals)
                max_val = max(all_vals)
                # If values differ significantly (spread > tolerance)
                if min_val > 0 and (max_val - min_val) / min_val > tolerance:
                    unique_vals = sorted(list(set(all_vals)))
                    sources = list(set([mc["source"] for mc in matched_chunks]))
                    conflicts.append({
                        "category": cat,
                        "values": unique_vals[:5],
                        "sources": sources
                    })

    return {
        "has_conflict": len(conflicts) > 0,
        "conflicts": conflicts
    }


def check_date_conflict(chunks: list) -> dict:
    """Identify if chunks mention conflicting/disparate policy or advisory years."""
    years_found = {}
    for c in chunks:
        text = c.get("text", "")
        matches = re.findall(r'\b(201[0-9]|202[0-9]|2030)\b', text)
        for y in matches:
            years_found.setdefault(y, []).append(c.get("source", "Document"))

    unique_years = sorted(list(years_found.keys()))
    if len(unique_years) > 1:
        y_min = int(unique_years[0])
        y_max = int(unique_years[-1])
        if (y_max - y_min) >= 2:
            details = (
                f"Referenced guidelines span multiple years ({', '.join(unique_years)}). "
                f"Older guidelines may reflect obsolete rates or past eligibility rules."
            )
            return {"has_conflict": True, "details": details}

    return {"has_conflict": False, "details": ""}


def run_conflict_check(query: str, chunks: list) -> dict:
    """Execute numeric and temporal consistency audits across retrieved candidate passages."""
    if not chunks:
        return {
            "has_conflict": False,
            "numeric_conflicts": [],
            "date_conflicts": {"has_conflict": False, "details": ""},
            "conflict_summary": ""
        }

    numeric = check_numeric_conflict(chunks)
    temporal = check_date_conflict(chunks)

    has_conflict = numeric["has_conflict"] or temporal["has_conflict"]

    summaries = []
    if numeric["has_conflict"]:
        categories = [c["category"] for c in numeric["conflicts"]]
        summaries.append(f"Discrepancies in recommended {', '.join(categories)}")
    if temporal["has_conflict"]:
        summaries.append("Cross-year document policy mismatch")

    return {
        "has_conflict": has_conflict,
        "numeric_conflicts": numeric["conflicts"],
        "date_conflicts": temporal,
        "conflict_summary": "; ".join(summaries)
    }
