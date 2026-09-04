"""Very small token accounting helpers.

Deliberately dependency-free: the research harness imports these to keep token
counting consistent between the fixture and the retrieval pipeline.
"""

CHARS_PER_TOKEN = 4


def estimate_tokens(text):
    """Rough token estimate for a string."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_to_budget(text, budget):
    """Truncate text so that estimate_tokens(result) <= budget."""
    if budget <= 0:
        return ""
    limit = budget * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    return text[:limit]


def fits(text, budget):
    return estimate_tokens(text) <= budget
