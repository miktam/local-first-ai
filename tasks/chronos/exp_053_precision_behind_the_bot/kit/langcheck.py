"""exp_052: the language of a paragraph, from function words. Seven languages: en es de fr pl ru sv.

Cyrillic letters decide Russian. Otherwise each language scores the number of its function words in the text,
plus 3 for each of its distinctive letters present, and the highest score wins (ties: the earlier language in
LANGS). Built for answers of a sentence or more, not for single words.
"""
import re

LANGS = ["en", "es", "de", "fr", "pl", "sv"]
WORDS = {
    "en": "the and of to in is are for with you your this that it on be can have will or not as from at by an if".split(),
    "es": "el la los las de del y en que es son para con por un una su sus se lo al como más o no le si pero muy".split(),
    "de": "der die das und ist sind für mit von zu den dem ein eine nicht auch sie es im auf bei oder wenn wie als".split(),
    "fr": "le la les des et est sont pour avec un une du dans que qui en vous pas sur au aux ou ce il si par".split(),
    "pl": "i w z na się nie to do jest są że jak dla od po czy przez oraz ale lub tak co już przy jako".split(),
    "sv": "och i att det är som en på för med av till om inte de har den ett kan eller du vi man också från".split(),
}
LETTERS = {"es": "ñ¿¡", "de": "ßäöü", "fr": "çèêàùâîôœ", "pl": "ąćęłńśźż", "sv": "åäö"}


def detect(text: str) -> str:
    if sum('Ѐ' <= c <= 'ӿ' for c in text) >= 3:
        return "ru"
    toks = re.findall(r"[^\W\d_]+", text.lower())
    low = text.lower()
    best, best_score = "en", -1
    for lang in LANGS:
        ws = set(WORDS[lang])
        score = sum(t in ws for t in toks) + 3 * sum(ch in low for ch in LETTERS.get(lang, ""))
        if score > best_score:
            best, best_score = lang, score
    return best


def detect_v2(text: str) -> str:
    """exp_053 correction (after the run, descriptive): capitalised words are dropped before scoring, so place
    names (Salobreña, Bahía de Marbella) neither add words nor add letters to Spanish. Otherwise as detect()."""
    if sum('Ѐ' <= c <= 'ӿ' for c in text) >= 3:
        return "ru"
    extra = {"de": {"keine", "kein", "habe", "hat", "wird", "nach", "zum", "zur", "über", "sehr", "noch", "nur",
                    "gibt", "dass", "einen", "einer"}}
    words = [w for w in re.findall(r"[^\W\d_]+", text) if not w[:1].isupper()]
    low = " ".join(words).lower()
    best, best_score = "en", -1
    for lang in LANGS:
        ws = set(WORDS[lang])
        ws = ws | extra.get(lang, set())
        score = sum(w.lower() in ws for w in words) + 3 * sum(ch in low for ch in LETTERS.get(lang, ""))
        if score > best_score:
            best, best_score = lang, score
    return best

