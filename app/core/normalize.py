import re


def normalize_name(name: str) -> str:
    """Canonical normalization used everywhere we compare device names.

    Strips everything except lowercase alphanumerics so that
    'OnePlus Nord CE5', 'oneplus-nord-ce5', and 'OnePlus Nord CE 5'
    all collapse to the same key. This key is stored in the indexed
    `normalized_name` column on Device / PhoneDirectory, so lookups are
    an exact-match on an index rather than a per-row regex computation.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())