"""Normalização morfológica em stdlib puro: stemming de Porter e lematização por regras.

O spec proíbe NLTK e o relatório gerado precisa continuar offline, então nem o
stemmer nem o lematizador podem depender de biblioteca externa ou de corpus
baixado em tempo de execução (WordNet, por exemplo). As duas funções públicas
são `stem` e `lemma`, ambas memoizadas: o corpus tem milhões de tokens mas
poucas centenas de milhares de formas distintas, e é a memoização que torna o
experimento viável.

Limite conhecido: sem léxico e sem etiquetas morfossintáticas, `lemma` faz
normalização flexional por regras mais uma tabela de irregulares. Isso não é
lematização no sentido do WordNet, que resolve `saw` para `see` ou `worse`
para `bad` consultando um dicionário e a classe gramatical. A tabela cobre os
irregulares frequentes em inglês; o resto recai nas regras.
"""

from __future__ import annotations

import re
from functools import lru_cache

VOWELS = "aeiou"
_WORD = re.compile(r"[a-z]+")

# Irregulares frequentes. Sem léxico, é o que separa `lemma` de um stemmer.
IRREGULAR: dict[str, str] = {
    "am": "be", "are": "be", "is": "be", "was": "be", "were": "be", "been": "be",
    "being": "be", "has": "have", "had": "have", "having": "have",
    "does": "do", "did": "do", "done": "do", "doing": "do",
    "went": "go", "gone": "go", "goes": "go",
    "said": "say", "says": "say", "made": "make", "makes": "make",
    "saw": "see", "seen": "see", "sees": "see", "took": "take", "taken": "take",
    "came": "come", "comes": "come", "got": "get", "gotten": "get",
    "gave": "give", "given": "give", "found": "find", "thought": "think",
    "knew": "know", "known": "know", "felt": "feel", "told": "tell",
    "became": "become", "left": "leave", "put": "put", "kept": "keep",
    "held": "hold", "meant": "mean", "met": "meet", "paid": "pay",
    "ran": "run", "sat": "sit", "spoke": "speak", "spoken": "speak",
    "stood": "stand", "understood": "understand", "wrote": "write",
    "written": "write", "lost": "lose", "sold": "sell", "sent": "send",
    "spent": "spend", "built": "build", "brought": "bring", "bought": "buy",
    "caught": "catch", "taught": "teach", "fought": "fight", "led": "lead",
    "fell": "fall", "fallen": "fall", "flew": "fly", "flown": "fly",
    "drew": "draw", "drawn": "draw", "grew": "grow", "grown": "grow",
    "began": "begin", "begun": "begin", "chose": "choose", "chosen": "choose",
    "broke": "break", "broken": "break", "wore": "wear", "worn": "wear",
    "shot": "shoot", "shown": "show", "showed": "show",
    # Comparativos e superlativos irregulares carregam sentimento, então valem
    # a entrada explícita: colapsá-los errado apagaria o sinal.
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "less": "little", "least": "little", "more": "much", "most": "much",
    "further": "far", "furthest": "far", "farther": "far", "farthest": "far",
    # Plurais irregulares.
    "children": "child", "men": "man", "women": "woman", "people": "person",
    "feet": "foot", "teeth": "tooth", "mice": "mouse", "geese": "goose",
    "lives": "life", "wives": "wife", "knives": "knife", "wolves": "wolf",
    "halves": "half", "selves": "self", "leaves": "leaf", "thieves": "thief",
    "shelves": "shelf", "loaves": "loaf", "calves": "calf",
    "criteria": "criterion", "phenomena": "phenomenon", "media": "medium",
    # O sufixo -ies é ambíguo e nenhuma regra resolve sem léxico: "studies"
    # vem de "study" (consoante + y -> ies), mas "movies" vem de "movie"
    # (-ie + s). A regra geral escolhe o padrão mais frequente, o -y, então os
    # substantivos em -ie precisam de entrada explícita — e neste corpus de
    # resenhas de filmes "movies" é dos tokens mais frequentes que existem.
    "movies": "movie", "cookies": "cookie", "zombies": "zombie",
    "rookies": "rookie", "pies": "pie", "ties": "tie", "lies": "lie",
    "dies": "die", "genies": "genie", "calories": "calorie",
    "prairies": "prairie", "junkies": "junkie", "newbies": "newbie",
    # Invariantes: a regra do -ies produziria "sery" e "specy".
    "series": "series", "species": "species",
}

# Sufixos -es que pertencem ao radical: sibilantes exigem o -es do plural.
_SIBILANT_END = ("s", "x", "z", "ch", "sh")


def _is_consonant(word: str, index: int) -> bool:
    letter = word[index]
    if letter in VOWELS:
        return False
    if letter != "y":
        return True
    # "y" é consoante no início e depois de vogal ("yes", "toy"), vogal depois
    # de consoante ("try").
    return index == 0 or not _is_consonant(word, index - 1)


def _measure(stem: str) -> int:
    """Conta as sequências vogal-consoante do radical (o `m` de Porter)."""
    count = 0
    previous_was_vowel = False
    for index in range(len(stem)):
        if _is_consonant(stem, index):
            if previous_was_vowel:
                count += 1
            previous_was_vowel = False
        else:
            previous_was_vowel = True
    return count


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, index) for index in range(len(stem)))


def _ends_with_double_consonant(stem: str) -> bool:
    return (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and _is_consonant(stem, len(stem) - 1)
    )


def _ends_cvc(stem: str) -> bool:
    """Consoante-vogal-consoante, com a última consoante fora de w, x, y."""
    if len(stem) < 3:
        return False
    return (
        _is_consonant(stem, len(stem) - 1)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 3)
        and stem[-1] not in "wxy"
    )


def _replace_if_measure(stem: str, suffix: str, replacement: str, minimum: int) -> str | None:
    if not stem.endswith(suffix):
        return None
    base = stem[: len(stem) - len(suffix)]
    if _measure(base) > minimum:
        return base + replacement
    return stem


_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
)
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def _step1a(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b_cleanup(stem: str) -> str:
    if stem.endswith(("at", "bl", "iz")):
        return stem + "e"
    if _ends_with_double_consonant(stem) and not stem.endswith(("l", "s", "z")):
        return stem[:-1]
    if _measure(stem) == 1 and _ends_cvc(stem):
        return stem + "e"
    return stem


def _step1b(word: str) -> str:
    if word.endswith("eed"):
        base = word[:-1]
        return base if _measure(word[:-3]) > 0 else word
    for suffix in ("ed", "ing"):
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if _contains_vowel(base):
                return _step1b_cleanup(base)
            return word
    return word


def _step1c(word: str) -> str:
    if word.endswith("y") and _contains_vowel(word[:-1]):
        return word[:-1] + "i"
    return word


def _step4(word: str) -> str:
    for suffix in _STEP4:
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if _measure(base) > 1:
                return base
            return word
    if word.endswith("ion"):
        base = word[:-3]
        if _measure(base) > 1 and base.endswith(("s", "t")):
            return base
    return word


def _step5(word: str) -> str:
    if word.endswith("e"):
        base = word[:-1]
        measure = _measure(base)
        if measure > 1 or (measure == 1 and not _ends_cvc(base)):
            word = base
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]
    return word


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    """Radical de Porter (1980). Palavras de até dois caracteres passam intactas."""
    if len(word) <= 2:
        return word
    result = _step1c(_step1b(_step1a(word)))
    for suffix, replacement in _STEP2:
        changed = _replace_if_measure(result, suffix, replacement, 0)
        if changed is not None:
            result = changed
            break
    for suffix, replacement in _STEP3:
        changed = _replace_if_measure(result, suffix, replacement, 0)
        if changed is not None:
            result = changed
            break
    return _step5(_step4(result))


def _depluralize(word: str) -> str | None:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2].endswith(_SIBILANT_END):
        return word[:-2]
    if word.endswith("es") and len(word) > 3:
        return word[:-1]
    if word.endswith("ss") or word.endswith("us") or word.endswith("is"):
        return None
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return None


def _deconjugate(word: str) -> str | None:
    for suffix in ("ing", "ed"):
        if not word.endswith(suffix) or len(word) - len(suffix) < 3:
            continue
        base = word[: -len(suffix)]
        if _ends_with_double_consonant(base) and not base.endswith(("l", "s", "z")):
            return base[:-1]
        # "hoping" -> "hope", mas "walking" -> "walk": o -e volta quando o
        # radical termina em consoante-vogal-consoante fraca.
        if _measure(base) == 1 and _ends_cvc(base):
            return base + "e"
        if base.endswith(("at", "iv", "iz", "bl", "us", "or", "ur", "ac")):
            return base + "e"
        return base
    return None


@lru_cache(maxsize=None)
def lemma(word: str) -> str:
    """Normalização flexional por regras, com tabela de irregulares.

    Não consulta léxico nem classe gramatical, então é mais conservadora que
    um lematizador do WordNet e mais conservadora que `stem`: só remove
    flexão (plural, tempo verbal, grau), nunca sufixos derivacionais.
    """
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) <= 3:
        return word
    for candidate in (_deconjugate(word), _depluralize(word)):
        if candidate and len(candidate) >= 2:
            return candidate
    if word.endswith("est") and len(word) > 5:
        return word[:-3]
    if word.endswith("er") and len(word) > 5:
        return word[:-2]
    return word


NORMALIZERS = {"none": None, "stemming": stem, "lemmatization": lemma}


def normalize_text(text: str, mode: str) -> str:
    """Aplica `mode` a cada token de um texto já limpo e em minúsculas."""
    if mode not in NORMALIZERS:
        raise ValueError(f"modo de normalização desconhecido: {mode}")
    normalizer = NORMALIZERS[mode]
    if normalizer is None:
        return text
    return " ".join(normalizer(token) for token in _WORD.findall(text))


def demo() -> None:
    """Verificação executável mínima das duas funções."""
    # Saídas finais do algoritmo completo, conferidas contra o vocabulário de
    # referência de Porter — não as intermediárias por passo do artigo de 1980.
    assert stem("running") == "run", stem("running")
    assert stem("studies") == "studi", stem("studies")
    assert stem("happiness") == "happi", stem("happiness")
    assert stem("relational") == "relat", stem("relational")
    assert stem("electriciti") == "electr", stem("electriciti")
    assert stem("hopping") == "hop", stem("hopping")
    assert stem("falling") == "fall", stem("falling")
    assert stem("is") == "is"

    assert lemma("running") == "run", lemma("running")
    assert lemma("studies") == "study", lemma("studies")
    assert lemma("movies") == "movie", lemma("movies")
    assert lemma("watched") == "watch", lemma("watched")
    assert lemma("worst") == "bad", lemma("worst")
    assert lemma("children") == "child"
    assert lemma("boring") == "bore", lemma("boring")
    # Lematização preserva derivação; stemming não. É o contraste do experimento.
    assert lemma("happiness") == "happiness", lemma("happiness")

    assert normalize_text("the movies were boring", "none") == "the movies were boring"
    assert normalize_text("the movies were boring", "lemmatization") == "the movie be bore"
    assert normalize_text("", "stemming") == ""
    print("text_normalization: todas as asserções passaram")


if __name__ == "__main__":
    demo()
