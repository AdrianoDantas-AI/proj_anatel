from __future__ import annotations

import csv
import html
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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
