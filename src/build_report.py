from __future__ import annotations

import csv
import html
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median, pstdev

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split


PROTECTED_NEGATIONS = {"not", "no", "nor", "never"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - PROTECTED_NEGATIONS
HTML_TAG = re.compile(r"<[^>]+>")
NON_LETTER = re.compile(r"[^a-z\s]")
WHITESPACE = re.compile(r"\s+")


def load_csv(
    path: Path, text_column: str, label_column: str
) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not path.is_file():
        raise ValueError(f"arquivo não encontrado: {path}")
    rows: list[dict[str, str]] = []
    malformed = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted({text_column, label_column} - columns)
        if missing:
            raise ValueError(f"colunas ausentes: {', '.join(missing)}")
        for row in reader:
            if None in row:
                malformed += 1
                continue
            rows.append(
                {
                    text_column: row.get(text_column) or "",
                    label_column: row.get(label_column) or "",
                }
            )
    return rows, {"malformed_rows": malformed}


def encode_rows(
    raw_rows: list[dict[str, str]],
    text_column: str,
    label_column: str,
    positive_label: str,
    negative_label: str,
) -> tuple[list[tuple[str, int]], dict[str, int]]:
    if positive_label == negative_label:
        raise ValueError("os rótulos positivo e negativo devem ser diferentes")
    expected = {positive_label, negative_label}
    unexpected = sorted(
        {
            row[label_column].strip()
            for row in raw_rows
            if row[label_column].strip() and row[label_column].strip() not in expected
        }
    )
    if unexpected:
        raise ValueError(f"rótulos inesperados: {', '.join(unexpected)}")

    encoded: list[tuple[str, int]] = []
    missing_text = missing_label = 0
    for row in raw_rows:
        text = row[text_column].strip()
        label = row[label_column].strip()
        if not text:
            missing_text += 1
            continue
        if not label:
            missing_label += 1
            continue
        encoded.append((row[text_column], int(label == positive_label)))
    return encoded, {"missing_text": missing_text, "missing_label": missing_label}


def clean_text(text: str) -> str:
    plain = html.unescape(HTML_TAG.sub(" ", text)).lower()
    letters = NON_LETTER.sub(" ", plain)
    tokens = (token for token in WHITESPACE.split(letters.strip()) if token)
    return " ".join(token for token in tokens if token not in STOPWORDS)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def profile_rows(rows: list[tuple[str, int]]) -> dict[str, object]:
    if not rows:
        raise ValueError("nenhuma linha válida para analisar")
    character_lengths = [len(text) for text, _ in rows]
    word_lengths = [len(text.split()) for text, _ in rows]
    labels = Counter(label for _, label in rows)
    by_class = {}
    samples = {}
    for label, name in ((0, "negative"), (1, "positive")):
        class_texts = [text for text, row_label in rows if row_label == label]
        class_characters = [len(text) for text in class_texts]
        class_words = [len(text.split()) for text in class_texts]
        by_class[name] = {
            "rows": len(class_texts),
            "median_characters": median(class_characters),
            "median_words": median(class_words),
        }
        samples[name] = class_texts[:3]
    return {
        "rows": len(rows),
        "class_counts": {
            "negative": labels[0],
            "positive": labels[1],
        },
        "html_rows": sum("<br" in text.lower() for text, _ in rows),
        "characters": {
            "mean": round(fmean(character_lengths), 2),
            "median": median(character_lengths),
            "p95": _percentile(character_lengths, 0.95),
            "min": min(character_lengths),
            "max": max(character_lengths),
        },
        "words": {
            "mean": round(fmean(word_lengths), 2),
            "median": median(word_lengths),
            "p95": _percentile(word_lengths, 0.95),
            "min": min(word_lengths),
            "max": max(word_lengths),
        },
        "by_class": by_class,
        "samples": samples,
    }


def deduplicate_rows(
    rows: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], dict[str, int]]:
    labels_by_text: dict[str, set[int]] = defaultdict(set)
    for text, label in rows:
        labels_by_text[text].add(label)
    conflicts = [text for text, labels in labels_by_text.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} textos duplicados possuem rótulos conflitantes"
        )
    counts = Counter(text for text, _ in rows)
    unique_rows = [(text, next(iter(labels))) for text, labels in labels_by_text.items()]
    duplicate_groups = sum(count > 1 for count in counts.values())
    return unique_rows, {
        "duplicate_groups": duplicate_groups,
        "duplicate_rows": len(rows) - len(unique_rows),
    }


def split_rows(
    rows: list[tuple[str, int]], random_state: int = 42
) -> dict[str, list[tuple[str, int]]]:
    labels = [label for _, label in rows]
    class_counts = Counter(labels)
    if set(class_counts) != {0, 1} or min(class_counts.values()) < 10:
        raise ValueError("cada classe precisa de pelo menos 10 textos únicos")
    train, remainder = train_test_split(
        rows,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )
    remainder_labels = [label for _, label in remainder]
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder_labels,
    )
    result = {"train": train, "validation": validation, "test": test}
    text_sets = {name: {text for text, _ in part} for name, part in result.items()}
    if not (
        text_sets["train"].isdisjoint(text_sets["validation"])
        and text_sets["train"].isdisjoint(text_sets["test"])
        and text_sets["validation"].isdisjoint(text_sets["test"])
    ):
        raise AssertionError("textos repetidos atravessaram os conjuntos")
    return result
