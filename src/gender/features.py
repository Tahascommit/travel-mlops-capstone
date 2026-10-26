"""Name handling for the gender model.

The model only looks at the first name. Age, company and travel behaviour were
checked in the notebook and carry no signal, so they are not used here.
"""
import re
import unicodedata

LABELS = ["female", "male"]


def first_name(full_name):
    """Lowercased, cleaned first token of a full name ('Maria Souza' -> 'maria').

    Same cleaning steps as the notebook: strip accents, lower case, drop punctuation
    and words with digits, collapse spaces, then keep the first word.
    """
    text = unicodedata.normalize("NFKD", str(full_name)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ")[0] if text else ""
