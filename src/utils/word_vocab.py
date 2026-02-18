"""
English word token ID set for restricting generation to word-like tokens.

Use with HuggingFace generate(..., prefix_allowed_tokens_fn=...) to avoid
code/special tokens (e.g. sp.ArgumentParser, {BR}) when generating next-word.
"""
from pathlib import Path
from typing import Optional, Set

# Small fallback list of common English words (tokenized to get allowed IDs)
_COMMON_WORDS = (
    "the and for are but not you all can had her was one our out day get has "
    "him his how man new now old see way who boy did its let put say she too "
    "use have this that with they from been more when will your said each "
    "them than then some into only other about just over very after where "
    "right still make most back through being much before under again "
    "confinement prison iranian kurdistan ultimately next question"
).split()


def get_english_word_token_ids(
    tokenizer,
    word_list_path: Optional[Path] = None,
    extra_words: Optional[list] = None,
) -> Set[int]:
    """
    Return set of token IDs that appear when tokenizing English words.

    Used to restrict generation to word tokens (e.g. prefix_allowed_tokens_fn)
    so the model cannot emit code/special tokens.

    Args:
        tokenizer: HuggingFace tokenizer (e.g. Llama).
        word_list_path: Optional path to a text file with one word per line.
        extra_words: Optional list of extra words to include (e.g. from dataset).

    Returns:
        Set of token IDs that are allowed when restricting to English words.
    """
    words = list(_COMMON_WORDS)
    if extra_words:
        words.extend(extra_words)
    if word_list_path and word_list_path.exists():
        with open(word_list_path) as f:
            words.extend(line.strip().lower() for line in f if line.strip())
    seen: Set[int] = set()
    for w in words:
        if not w:
            continue
        ids = tokenizer.encode(w, add_special_tokens=False)
        seen.update(ids)
    return seen
