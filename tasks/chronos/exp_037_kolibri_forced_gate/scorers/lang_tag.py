"""Frozen deterministic language tagger: "de" | "en" | "mixed" | "unknown" (BUILD_SPEC §5.6).

Used by the gate's behaviour check (G5: >= 2/20 "unknown" blocks) and by E4 (the share of German
reasoning segments on German items). No model and no external package: two frozen stopword lists of
about 100 function words each (108 English, 107 German), chosen to be unambiguous between the two
languages. Words common in both, such as "in", "an", "so", "was", "will", "also", "man", "war" and
"her", are in neither list.

Rule (frozen):
1. Lower-case the text and take the runs of letters a-z plus ä ö ü ß as words.
2. n_de = words in GERMAN, n_en = words in ENGLISH, n = n_de + n_en.
3. n < MIN_HITS -> "unknown" (too few function words to tell: empty text, numbers, code, symbols,
   another script, or a one- or two-word reply).
4. share = n_de / n.  share >= DE_SHARE -> "de";  share <= EN_SHARE -> "en";  otherwise "mixed".
"""

from __future__ import annotations

import re

MIN_HITS = 2
DE_SHARE = 0.8
EN_SHARE = 0.2

ENGLISH = frozenset("""
the and of to is are be been being that this these those it its for with on at by from or not but
what which who whom whose when where why how all any each every some many most more much other such
only very just than then there here they them their theirs we our ours you your yours he him his she
hers my me mine i if because while after before through between into over under about against
during without would could should can may might must shall does did do done has have had having
were yet both either neither nor though although
""".split())

GERMAN = frozenset("""
der die das den dem des ein eine einen einem einer eines und ist sind nicht zu mit sich auf für von
sie es auch dass wie aus bei nach oder aber wenn noch nur werden wird wurde wurden kann können hat
haben habe hatte über unter durch um im vom zum zur dieser diese dieses diesem diesen sein seine
ihre ihr wir ich du er schon sehr mehr immer dann doch weil bis ohne gegen zwischen während hier
dort alle keine kein jetzt sowie bereits beim damit dabei denn gibt ob welche welcher welches warum
uns ihnen ihm ihn mir mich nichts sondern worden sollte muss
""".split())

_WORD = re.compile(r"[a-zäöüß]+")


def counts(text: str) -> tuple[int, int]:
    """(n_de, n_en): German and English stopword hits in the text."""
    words = _WORD.findall(text.lower())
    n_de = sum(1 for w in words if w in GERMAN)
    n_en = sum(1 for w in words if w in ENGLISH)
    return n_de, n_en


def tag(text: str | None) -> str:
    """'de' | 'en' | 'mixed' | 'unknown' by the frozen rule in the module docstring."""
    if not text:
        return "unknown"
    n_de, n_en = counts(text)
    n = n_de + n_en
    if n < MIN_HITS:
        return "unknown"
    share = n_de / n
    if share >= DE_SHARE:
        return "de"
    if share <= EN_SHARE:
        return "en"
    return "mixed"
