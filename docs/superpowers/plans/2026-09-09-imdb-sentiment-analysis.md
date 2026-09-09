# IMDb Sentiment Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible IMDb sentiment-analysis pipeline and a single offline HTML report that documents the complete study and performs local EDA on user-selected CSV files.

**Architecture:** One Python entry point owns validation, EDA, leakage-safe splitting, Bag of Words modeling, evaluation, and report-data generation. A separate HTML template owns presentation and browser-only CSV analysis; the build inlines its CSS, JavaScript, report payload, and pinned Papa Parse source into one portable output file.

**Tech Stack:** Python 3.14, Python standard library, scikit-learn 1.9, pytest, HTML5, CSS, vanilla JavaScript, Papa Parse 5.5.3

**Spec:** `docs/superpowers/specs/2026-09-09-imdb-sentiment-analysis-design.md`

## Global Constraints

- Preserve `IMDB%20Dataset.csv` byte-for-byte and keep it out of Git.
- Use exact-text deduplication before a stratified 70/15/15 split with `random_state=42`.
- Fit cleaning-dependent vocabulary and models only on the training partition.
- Use Bag of Words with `(1, 2)` n-grams, `min_df=2`, and `max_features=50_000`.
- Preserve `not`, `no`, `nor`, and `never` while removing other English stopwords.
- Use Multinomial Naive Bayes as baseline and Logistic Regression as the comparison model.
- Select the cross-validation candidate from validation F1 before evaluating the frozen models on test.
- Keep the generated HTML offline, self-contained, and free of network requests.
- New CSVs selected in the browser receive EDA only; they never train or invoke a model.
- Use no pandas, NLTK, template engine, chart library, backend, database, or model persistence.
- Treat every Git commit step below as an explicit human-approval gate; never commit merely because the step is reached.

---

## File Map

- `.gitignore`: excludes the raw CSV, virtual environment, Python caches, and pytest cache.
- `requirements.txt`: pins the one runtime dependency and bounds the test dependency.
- `src/build_report.py`: owns the complete Python pipeline and command-line entry point.
- `src/report_template.html`: owns layout, visual behavior, embedded documentation, and browser-side EDA.
- `src/vendor/papaparse.min.js`: pinned CSV parser that is inlined into the generated report.
- `tests/test_pipeline.py`: contains the focused runnable checks for the Python and generated HTML contracts.
- `reports/imdb_analysis.html`: generated, portable interview deliverable.

---

### Task 1: Data Contract and Text Cleaning

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `src/build_report.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `load_csv(path: Path, text_column: str, label_column: str) -> tuple[list[dict[str, str]], dict[str, int]]`
- Produces: `encode_rows(raw_rows: list[dict[str, str]], text_column: str, label_column: str, positive_label: str, negative_label: str) -> tuple[list[tuple[str, int]], dict[str, int]]`
- Produces: `clean_text(text: str) -> str`

- [ ] **Step 1: Add the isolated environment and raw-data exclusions**

Create `.gitignore` with exactly:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
IMDB%20Dataset.csv
```

Create `requirements.txt` with:

```text
scikit-learn==1.9.0
pytest>=9,<10
```

- [ ] **Step 2: Create and populate the virtual environment**

Run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: all commands exit `0`; `scikit-learn 1.9.0` and a pytest release in the `9.x` series are installed only under `.venv`.

- [ ] **Step 3: Write failing tests for loading, validation, and cleaning**

Create `tests/test_pipeline.py` with:

```python
from pathlib import Path

import pytest

from src.build_report import clean_text, encode_rows, load_csv


def write_csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "reviews.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_encode_and_clean_text(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        'review,sentiment\n"<br>NOT a bad movie! 10/10 &amp; never boring.",positive\n'
        '"Awful...",negative\n"",positive\n',
    )

    raw_rows, load_stats = load_csv(path, "review", "sentiment")
    rows, quality = encode_rows(
        raw_rows, "review", "sentiment", "positive", "negative"
    )

    assert load_stats == {"malformed_rows": 0}
    assert rows == [
        ("<br>NOT a bad movie! 10/10 &amp; never boring.", 1),
        ("Awful...", 0),
    ]
    assert quality == {"missing_text": 1, "missing_label": 0}
    assert clean_text(rows[0][0]) == "not bad movie never boring"


def test_rejects_missing_columns_and_unexpected_labels(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "text,label\nGood,maybe\n")

    with pytest.raises(ValueError, match="colunas ausentes"):
        load_csv(path, "review", "sentiment")

    raw_rows, _ = load_csv(path, "text", "label")
    with pytest.raises(ValueError, match="rótulos inesperados: maybe"):
        encode_rows(raw_rows, "text", "label", "positive", "negative")
```

- [ ] **Step 4: Run the tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'src.build_report'`.

- [ ] **Step 5: Implement the minimum data contract**

Create `src/build_report.py` with these imports, constants, and functions:

```python
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
```

- [ ] **Step 6: Run Task 1 tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Expected: `2 passed` and exit code `0`.

- [ ] **Step 7: Review the Task 1 diff and request commit approval**

Run:

```powershell
git diff -- .gitignore requirements.txt src\build_report.py tests\test_pipeline.py
git status --short
```

Expected: the four Task 1 files are untracked or modified; `IMDB%20Dataset.csv` is absent from status because `.gitignore` excludes it.

After explicit approval only:

```powershell
git add .gitignore requirements.txt src\build_report.py tests\test_pipeline.py
git commit -m "feat: add sentiment data contract"
```

---

### Task 2: EDA, Deduplication, and Leakage-Safe Splits

**Files:**
- Modify: `src/build_report.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: encoded rows as `list[tuple[str, int]]`
- Produces: `profile_rows(rows: list[tuple[str, int]]) -> dict[str, object]`
- Produces: `deduplicate_rows(rows: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], dict[str, int]]`
- Produces: `split_rows(rows: list[tuple[str, int]], random_state: int = 42) -> dict[str, list[tuple[str, int]]]`

- [ ] **Step 1: Append failing EDA and split tests**

Append to `tests/test_pipeline.py`:

```python
from src.build_report import deduplicate_rows, profile_rows, split_rows


def make_balanced_rows(count_per_class: int = 20) -> list[tuple[str, int]]:
    return [
        *((f"good movie number {index}", 1) for index in range(count_per_class)),
        *((f"bad movie number {index}", 0) for index in range(count_per_class)),
    ]


def test_profiles_and_deduplicates_before_split() -> None:
    rows = make_balanced_rows()
    rows.extend([rows[0], rows[-1]])

    profile = profile_rows(rows)
    unique_rows, duplicate_stats = deduplicate_rows(rows)
    splits = split_rows(unique_rows)

    assert profile["rows"] == 42
    assert profile["class_counts"] == {"negative": 21, "positive": 21}
    assert profile["by_class"]["negative"]["rows"] == 21
    assert profile["by_class"]["positive"]["rows"] == 21
    assert len(profile["samples"]["negative"]) == 3
    assert len(profile["samples"]["positive"]) == 3
    assert duplicate_stats == {"duplicate_groups": 2, "duplicate_rows": 2}
    assert {name: len(part) for name, part in splits.items()} == {
        "train": 28,
        "validation": 6,
        "test": 6,
    }
    text_sets = {name: {text for text, _ in part} for name, part in splits.items()}
    assert text_sets["train"].isdisjoint(text_sets["validation"])
    assert text_sets["train"].isdisjoint(text_sets["test"])
    assert text_sets["validation"].isdisjoint(text_sets["test"])
    assert all(sum(label for _, label in part) == len(part) // 2 for part in splits.values())


def test_rejects_duplicate_text_with_conflicting_labels() -> None:
    with pytest.raises(ValueError, match="rótulos conflitantes"):
        deduplicate_rows([("same review", 1), ("same review", 0)])
```

- [ ] **Step 2: Run the new tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py::test_profiles_and_deduplicates_before_split tests\test_pipeline.py::test_rejects_duplicate_text_with_conflicting_labels -v
```

Expected: collection fails because `profile_rows`, `deduplicate_rows`, and `split_rows` do not exist.

- [ ] **Step 3: Implement EDA, deduplication, and split functions**

Add imports:

```python
from collections import Counter, defaultdict
from statistics import fmean, median, pstdev

from sklearn.model_selection import train_test_split
```

Add:

```python
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
```

- [ ] **Step 4: Run all tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Expected: `4 passed` and exit code `0`.

- [ ] **Step 5: Review the Task 2 diff and request commit approval**

Run:

```powershell
git diff -- src\build_report.py tests\test_pipeline.py
```

After explicit approval only:

```powershell
git add src\build_report.py tests\test_pipeline.py
git commit -m "feat: add leakage-safe sentiment EDA"
```

---

### Task 3: Bag of Words Models and Evaluation

**Files:**
- Modify: `src/build_report.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: output from `split_rows`
- Produces: `make_model(kind: str) -> Pipeline`
- Produces: `binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]`
- Produces: `run_models(splits: dict[str, list[tuple[str, int]]]) -> tuple[dict[str, Pipeline], dict[str, object]]`

- [ ] **Step 1: Append the failing modeling test**

Append:

```python
from src.build_report import run_models


def test_models_use_train_vocabulary_and_return_required_metrics() -> None:
    train = [
        *((f"excellent warm story sharedword {index}", 1) for index in range(12)),
        *((f"awful cold story sharedword {index}", 0) for index in range(12)),
    ]
    validation = [
        *((f"excellent warm validation {index}", 1) for index in range(5)),
        *((f"awful cold validation {index}", 0) for index in range(5)),
    ]
    test = [
        *((f"excellent warm testonlyword {index}", 1) for index in range(5)),
        *((f"awful cold testonlyword {index}", 0) for index in range(5)),
    ]

    models, result = run_models(
        {"train": train, "validation": validation, "test": test}
    )

    assert set(models) == {"naive_bayes", "logistic_regression"}
    assert "testonlyword" not in models["naive_bayes"].named_steps["bow"].vocabulary_
    assert result["selected_model"] in models
    assert len(result["cross_validation_f1"]) == 5
    for model_result in result["models"].values():
        assert set(model_result["validation"]["metrics"]) == {
            "accuracy", "precision", "recall", "f1", "confusion_matrix"
        }
        assert set(model_result["test"]["metrics"]) == {
            "accuracy", "precision", "recall", "f1", "confusion_matrix"
        }
```

- [ ] **Step 2: Run the modeling test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py::test_models_use_train_vocabulary_and_return_required_metrics -v
```

Expected: collection fails because `run_models` does not exist.

- [ ] **Step 3: Implement fixed model pipelines and metrics**

Add imports:

```python
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
```

Add:

```python
def make_model(kind: str) -> Pipeline:
    bow = CountVectorizer(
        preprocessor=clean_text,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
    )
    if kind == "naive_bayes":
        classifier = MultinomialNB()
    elif kind == "logistic_regression":
        classifier = LogisticRegression(max_iter=1_000, random_state=42)
    else:
        raise ValueError(f"modelo desconhecido: {kind}")
    return Pipeline([("bow", bow), ("classifier", classifier)])


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _xy(rows: list[tuple[str, int]]) -> tuple[list[str], list[int]]:
    return [text for text, _ in rows], [label for _, label in rows]


def _error_samples(
    rows: list[tuple[str, int]], predictions: list[int], limit: int = 5
) -> dict[str, list[dict[str, object]]]:
    false_positives = []
    false_negatives = []
    for (text, expected), predicted in zip(rows, predictions, strict=True):
        if expected == 0 and predicted == 1 and len(false_positives) < limit:
            false_positives.append({"text": text[:700], "expected": 0, "predicted": 1})
        if expected == 1 and predicted == 0 and len(false_negatives) < limit:
            false_negatives.append({"text": text[:700], "expected": 1, "predicted": 0})
    return {"false_positives": false_positives, "false_negatives": false_negatives}


def run_models(
    splits: dict[str, list[tuple[str, int]]]
) -> tuple[dict[str, Pipeline], dict[str, object]]:
    train_x, train_y = _xy(splits["train"])
    validation_x, validation_y = _xy(splits["validation"])
    test_x, test_y = _xy(splits["test"])
    models: dict[str, Pipeline] = {}
    results: dict[str, Any] = {}

    for kind in ("naive_bayes", "logistic_regression"):
        model = make_model(kind)
        model.fit(train_x, train_y)
        validation_predictions = model.predict(validation_x).tolist()
        models[kind] = model
        results[kind] = {
            "validation": {"metrics": binary_metrics(validation_y, validation_predictions)},
        }

    selected = max(results, key=lambda name: results[name]["validation"]["metrics"]["f1"])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(make_model(selected), train_x, train_y, cv=folds, scoring="f1")

    for kind, model in models.items():
        test_predictions = model.predict(test_x).tolist()
        results[kind]["test"] = {
            "metrics": binary_metrics(test_y, test_predictions),
            "errors": _error_samples(splits["test"], test_predictions),
        }

    return models, {
        "selected_model": selected,
        "cross_validation_f1": [round(float(score), 4) for score in scores],
        "cross_validation_mean": round(fmean(float(score) for score in scores), 4),
        "cross_validation_std": round(pstdev(float(score) for score in scores), 4),
        "models": results,
    }
```

- [ ] **Step 4: Run all tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Expected: `5 passed` and exit code `0`.

- [ ] **Step 5: Review the Task 3 diff and request commit approval**

Run:

```powershell
git diff -- src\build_report.py tests\test_pipeline.py
```

After explicit approval only:

```powershell
git add src\build_report.py tests\test_pipeline.py
git commit -m "feat: compare bag of words classifiers"
```

---

### Task 4: Offline Report Rendering

**Files:**
- Create: `src/report_template.html`
- Modify: `src/build_report.py`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: serializable analysis payload returned by Task 3
- Produces: `render_report(payload: dict[str, object], template_path: Path, papa_path: Path, output_path: Path) -> None`
- Template contract: placeholders `__PAPA_PARSE_SOURCE__` and `__REPORT_DATA__` occur exactly once and are absent from generated output.

- [ ] **Step 1: Append a failing render contract test**

Append:

```python
from src.build_report import render_report


REQUIRED_SECTION_IDS = {
    "summary", "method", "pipeline", "eda", "leakage",
    "models", "errors", "decisions", "conclusion", "local-csv"
}


def test_rendered_report_is_self_contained(tmp_path: Path) -> None:
    template = tmp_path / "template.html"
    template.write_text(
        '<!doctype html><html><body><script>__PAPA_PARSE_SOURCE__</script>'
        '<script>const REPORT = __REPORT_DATA__;</script>'
        + "".join(f'<section id="{name}"></section>' for name in REQUIRED_SECTION_IDS)
        + "</body></html>",
        encoding="utf-8",
    )
    papa = tmp_path / "papa.js"
    papa.write_text("window.Papa = {};", encoding="utf-8")
    output = tmp_path / "report.html"

    render_report({"title": "IMDb </script> safe"}, template, papa, output)
    rendered = output.read_text(encoding="utf-8")

    assert "__REPORT_DATA__" not in rendered
    assert "__PAPA_PARSE_SOURCE__" not in rendered
    assert "<\\/script>" in rendered
    assert 'src="http' not in rendered
    assert 'href="http' not in rendered
    assert all(f'id="{name}"' in rendered for name in REQUIRED_SECTION_IDS)
```

- [ ] **Step 2: Run the render test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py::test_rendered_report_is_self_contained -v
```

Expected: collection fails because `render_report` does not exist.

- [ ] **Step 3: Implement safe, atomic template rendering**

Add imports:

```python
import json
import os
import tempfile
```

Add:

```python
def render_report(
    payload: dict[str, object],
    template_path: Path,
    papa_path: Path,
    output_path: Path,
) -> None:
    template = template_path.read_text(encoding="utf-8")
    if template.count("__REPORT_DATA__") != 1 or template.count("__PAPA_PARSE_SOURCE__") != 1:
        raise ValueError("template deve conter cada placeholder exatamente uma vez")
    safe_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    rendered = template.replace(
        "__PAPA_PARSE_SOURCE__", papa_path.read_text(encoding="utf-8")
    ).replace("__REPORT_DATA__", safe_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, suffix=".html", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(rendered)
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
```

- [ ] **Step 4: Build the semantic HTML shell**

Create `src/report_template.html` with one `<main>`, the ten required section IDs, a skip link, descriptive headings, and these embedded data anchors:

```html
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Análise de Sentimentos IMDb</title>
  <style>
    :root { color-scheme: dark; --bg:#08111f; --panel:#111d30; --text:#edf4ff; --muted:#9fb0c8; --positive:#39d98a; --negative:#ff6b78; --accent:#65a8ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:16px/1.6 system-ui,sans-serif; }
    main { width:min(1120px,92vw); margin:auto; }
    section { padding:4rem 0; }
    .card { background:var(--panel); border:1px solid #263650; border-radius:18px; padding:1.25rem; }
    .metric { font-size:clamp(1.8rem,5vw,3.5rem); font-weight:750; }
    .sr-only { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0,0,0,0); }
    .sr-only:focus { position:fixed; width:auto; height:auto; clip:auto; top:1rem; left:1rem; padding:.75rem; background:var(--panel); z-index:10; }
    @media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms!important; transition-duration:.01ms!important; } }
  </style>
</head>
<body>
  <a class="sr-only" href="#summary">Ir para o conteúdo</a>
  <main>
    <section id="summary"><h1>Análise de Sentimentos IMDb</h1></section>
    <section id="method"><h2>Objetivo e metodologia</h2></section>
    <section id="pipeline"><h2>Do CSV à decisão</h2></section>
    <section id="eda"><h2>Qualidade dos dados</h2></section>
    <section id="leakage"><h2>Split e prevenção de vazamento</h2></section>
    <section id="models"><h2>Comparação dos modelos</h2></section>
    <section id="errors"><h2>Onde os modelos erram</h2></section>
    <section id="decisions"><h2>Decisões técnicas</h2></section>
    <section id="conclusion"><h2>Conclusões e próximos passos</h2></section>
    <section id="local-csv"><h2>Analise outro CSV localmente</h2></section>
  </main>
  <script>__PAPA_PARSE_SOURCE__</script>
  <script>const REPORT = __REPORT_DATA__;</script>
</body>
</html>
```

Expand each section in this same file with labeled containers for the exact payload fields; do not duplicate prose already stored in the `decisions` payload. Use native bars, grids, and tables rather than adding a chart dependency.

- [ ] **Step 5: Run all tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Expected: `6 passed` and exit code `0`.

- [ ] **Step 6: Review the Task 4 diff and request commit approval**

Run:

```powershell
git diff -- src\build_report.py src\report_template.html tests\test_pipeline.py
```

After explicit approval only:

```powershell
git add src\build_report.py src\report_template.html tests\test_pipeline.py
git commit -m "feat: render offline analysis report"
```

---

### Task 5: Browser-Only CSV EDA and Consumption Animation

**Files:**
- Create: `src/vendor/papaparse.min.js`
- Modify: `src/report_template.html`
- Modify: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: a local `File` selected through `<input type="file" accept=".csv,text/csv">`
- Produces in JavaScript: `summarizeRows(rows, config) -> summary`
- Produces in JavaScript: `readLabels(file, labelColumn) -> Promise<string[]>`
- Produces in JavaScript: `analyzeLocalCsv(file, config) -> Promise<summary>`
- Updates only elements under `#local-csv`; never sends data or calls the Python model.

- [ ] **Step 1: Vendor the exact parser release**

Run:

```powershell
New-Item -ItemType Directory -Force src\vendor | Out-Null
Invoke-WebRequest -Uri "https://cdn.jsdelivr.net/npm/papaparse@5.5.3/papaparse.min.js" -OutFile "src\vendor\papaparse.min.js"
Get-Item src\vendor\papaparse.min.js | Select-Object Name,Length
```

Expected: exit code `0`, filename `papaparse.min.js`, and non-zero length. Inspect the header to confirm it identifies Papa Parse 5.5.3 before proceeding.

- [ ] **Step 2: Add a browser-contract test against the real template**

Append:

```python
def test_real_template_contains_offline_csv_analyzer(tmp_path: Path) -> None:
    output = tmp_path / "real-report.html"
    render_report(
        {},
        Path("src/report_template.html"),
        Path("src/vendor/papaparse.min.js"),
        output,
    )
    rendered = output.read_text(encoding="utf-8")

    for identifier in (
        "csv-file", "text-column", "label-column", "positive-label",
        "negative-label", "scan-labels", "analyze-csv", "csv-progress",
        "csv-error", "csv-results"
    ):
        assert f'id="{identifier}"' in rendered
    assert "function summarizeRows" in rendered
    assert "function runBrowserSelfCheck" in rendered
    assert "fetch(" not in rendered
    assert "XMLHttpRequest" not in rendered
```

- [ ] **Step 3: Run the browser-contract test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py::test_real_template_contains_offline_csv_analyzer -v
```

Expected: failure on the first missing browser control ID.

- [ ] **Step 4: Add the explicit two-pass CSV workflow**

Inside `#local-csv`, add:

```html
<div class="card">
  <label for="csv-file">Selecione um CSV</label>
  <input id="csv-file" type="file" accept=".csv,text/csv">
  <label for="text-column">Coluna de texto</label><select id="text-column" disabled></select>
  <label for="label-column">Coluna de classe</label><select id="label-column" disabled></select>
  <button id="scan-labels" type="button" disabled>Ler classes</button>
  <label for="positive-label">Valor positivo</label><select id="positive-label" disabled></select>
  <label for="negative-label">Valor negativo</label><select id="negative-label" disabled></select>
  <button id="analyze-csv" type="button" disabled>Analisar CSV</button>
  <progress id="csv-progress" max="100" value="0"></progress>
  <p id="csv-error" role="alert" hidden></p>
  <div id="csv-results" aria-live="polite"></div>
</div>
```

Add JavaScript with one shared aggregation path for the self-check and streaming parser:

```javascript
function emptySummary() {
  return {rows:0, valid:0, invalid:0, missingText:0, missingLabel:0,
    unexpectedLabel:0, positive:0, negative:0, duplicates:0, htmlRows:0};
}

function consumeRow(summary, row, config, seen, lengths) {
  summary.rows += 1;
  const text = String(row[config.textColumn] ?? "").trim();
  const label = String(row[config.labelColumn] ?? "").trim();
  if (!text) { summary.missingText += 1; summary.invalid += 1; return; }
  if (!label) { summary.missingLabel += 1; summary.invalid += 1; return; }
  if (label !== config.positiveLabel && label !== config.negativeLabel) {
    summary.unexpectedLabel += 1; summary.invalid += 1; return;
  }
  summary.valid += 1;
  summary[label === config.positiveLabel ? "positive" : "negative"] += 1;
  summary.htmlRows += /<br\s*\/?\s*>/i.test(text) ? 1 : 0;
  summary.duplicates += seen.has(text) ? 1 : 0;
  seen.add(text);
  lengths.characters.push(text.length);
  lengths.words.push(text.split(/\s+/).length);
}

function finishSummary(summary, lengths) {
  lengths.characters.sort((a, b) => a - b);
  lengths.words.sort((a, b) => a - b);
  const count = lengths.characters.length;
  summary.meanCharacters = count
    ? Math.round(lengths.characters.reduce((total, value) => total + value, 0) / count)
    : 0;
  summary.medianCharacters = count ? lengths.characters[Math.floor(count / 2)] : 0;
  summary.p95Characters = count ? lengths.characters[Math.floor((count - 1) * .95)] : 0;
  summary.medianWords = count ? lengths.words[Math.floor(count / 2)] : 0;
  return summary;
}

function summarizeRows(rows, config) {
  const summary = emptySummary(), seen = new Set(), lengths = {characters:[], words:[]};
  for (const row of rows) consumeRow(summary, row, config, seen, lengths);
  return finishSummary(summary, lengths);
}

function updateProgress(cursor, file) {
  document.querySelector("#csv-progress").value = Math.min(100, Math.round(cursor / file.size * 100));
}

function readLabels(file, labelColumn) {
  return new Promise((resolve, reject) => {
    const labels = new Set();
    let settled = false;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      chunk(results, parser) {
        const blocking = results.errors.find(error =>
          error.code === "MissingQuotes" || error.code === "UndetectableDelimiter");
        if (blocking) {
          settled = true;
          parser.abort();
          reject(new Error(`CSV inválido: ${blocking.message}`));
          return;
        }
        for (const row of results.data) {
          const label = String(row[labelColumn] ?? "").trim();
          if (label) labels.add(label);
          if (labels.size > 100) {
            settled = true;
            parser.abort();
            reject(new Error("A coluna escolhida possui mais de 100 valores; selecione uma coluna de classe."));
            return;
          }
        }
        updateProgress(results.meta.cursor, file);
      },
      complete() { if (!settled) resolve([...labels].sort()); },
      error(error) { if (!settled) reject(error); }
    });
  });
}

function analyzeLocalCsv(file, config) {
  if (config.textColumn === config.labelColumn) {
    return Promise.reject(new Error("Escolha colunas diferentes para texto e classe."));
  }
  if (config.positiveLabel === config.negativeLabel) {
    return Promise.reject(new Error("Escolha valores diferentes para as duas classes."));
  }
  return new Promise((resolve, reject) => {
    const summary = emptySummary(), seen = new Set(), lengths = {characters:[], words:[]};
    let settled = false;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      chunk(results, parser) {
        const blocking = results.errors.find(error =>
          error.code === "MissingQuotes" || error.code === "UndetectableDelimiter");
        if (blocking) {
          settled = true;
          parser.abort();
          reject(new Error(`CSV inválido: ${blocking.message}`));
          return;
        }
        for (const row of results.data) consumeRow(summary, row, config, seen, lengths);
        updateProgress(results.meta.cursor, file);
      },
      complete() {
        if (settled) return;
        finishSummary(summary, lengths);
        if (!summary.positive || !summary.negative) {
          reject(new Error("O CSV precisa conter exemplos válidos das duas classes selecionadas."));
        } else {
          resolve(summary);
        }
      },
      error(error) { if (!settled) reject(error); }
    });
  });
}

function runBrowserSelfCheck() {
  const result = summarizeRows(
    [{text:"Good<br />film", label:"yes"}, {text:"Bad", label:"no"},
     {text:"Bad", label:"no"}, {text:"", label:"yes"}],
    {textColumn:"text", labelColumn:"label", positiveLabel:"yes", negativeLabel:"no"}
  );
  console.assert(result.rows === 4 && result.duplicates === 1 && result.htmlRows === 1 && result.missingText === 1,
    "browser EDA self-check failed");
}
runBrowserSelfCheck();
```

Wire the controls with `addEventListener`. On file selection, call `Papa.parse(file, {header:true, preview:1})`, read `results.meta.fields`, and populate both column selectors with `document.createElement("option")` plus `textContent`. The **Ler classes** handler calls `readLabels` and populates both explicit label selectors. The **Analisar CSV** handler builds the four-field config, awaits `analyzeLocalCsv`, and renders rows, validity, each missing-value count, class proportions, duplicates, HTML presence, and median size by creating DOM nodes and assigning `textContent`. On every failure, clear the previous result, reveal `#csv-error`, and keep its message actionable.

```javascript
const controls = Object.fromEntries([
  "csv-file", "text-column", "label-column", "positive-label", "negative-label",
  "scan-labels", "analyze-csv", "csv-error", "csv-results"
].map(id => [id, document.getElementById(id)]));
let selectedFile = null;

function fillSelect(select, values) {
  select.replaceChildren(...values.map(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    return option;
  }));
  select.disabled = values.length === 0;
}

function showError(error) {
  controls["csv-results"].replaceChildren();
  controls["csv-error"].textContent = error.message || String(error);
  controls["csv-error"].hidden = false;
}

function renderLocalSummary(summary) {
  controls["csv-error"].hidden = true;
  const labels = {
    rows:"Linhas", valid:"Válidas", invalid:"Inválidas", missingText:"Sem texto",
    missingLabel:"Sem classe", unexpectedLabel:"Classe inesperada", positive:"Positivas",
    negative:"Negativas", duplicates:"Duplicadas", htmlRows:"Com HTML",
    meanCharacters:"Média de caracteres", medianCharacters:"Mediana de caracteres",
    p95Characters:"P95 de caracteres", medianWords:"Mediana de palavras"
  };
  controls["csv-results"].replaceChildren(...Object.entries(labels).map(([key, label]) => {
    const card = document.createElement("div");
    card.className = "card metric-card";
    const title = document.createElement("span");
    const value = document.createElement("strong");
    title.textContent = label;
    value.textContent = String(summary[key]);
    card.append(title, value);
    return card;
  }));
}

controls["csv-file"].addEventListener("change", () => {
  selectedFile = controls["csv-file"].files[0] || null;
  controls["scan-labels"].disabled = true;
  if (!selectedFile) return;
  Papa.parse(selectedFile, {
    header:true, preview:1,
    complete(results) {
      const blocking = results.errors.find(error =>
        error.code === "MissingQuotes" || error.code === "UndetectableDelimiter");
      if (blocking) return showError(new Error(`CSV inválido: ${blocking.message}`));
      const fields = results.meta.fields || [];
      if (fields.length < 2) return showError(new Error("O CSV precisa ter ao menos duas colunas."));
      fillSelect(controls["text-column"], fields);
      fillSelect(controls["label-column"], fields);
      controls["scan-labels"].disabled = false;
    },
    error:showError
  });
});

controls["scan-labels"].addEventListener("click", async () => {
  try {
    const labels = await readLabels(selectedFile, controls["label-column"].value);
    if (labels.length < 2) throw new Error("A coluna de classe precisa ter ao menos dois valores.");
    fillSelect(controls["positive-label"], labels);
    fillSelect(controls["negative-label"], labels);
    controls["analyze-csv"].disabled = false;
  } catch (error) { showError(error); }
});

controls["label-column"].addEventListener("change", () => {
  fillSelect(controls["positive-label"], []);
  fillSelect(controls["negative-label"], []);
  controls["analyze-csv"].disabled = true;
});

controls["analyze-csv"].addEventListener("click", async () => {
  try {
    renderLocalSummary(await analyzeLocalCsv(selectedFile, {
      textColumn:controls["text-column"].value,
      labelColumn:controls["label-column"].value,
      positiveLabel:controls["positive-label"].value,
      negativeLabel:controls["negative-label"].value
    }));
  } catch (error) { showError(error); }
});
```

- [ ] **Step 5: Add restrained motion and accessible state changes**

Use CSS transitions for stage activation, metric count-up, and proportional bars. Bind animation state to actual parser callbacks. Keep final values visible without motion, preserve keyboard focus outlines, use `role="alert"` for failures, and use `aria-live="polite"` for completed results.

- [ ] **Step 6: Run all tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Expected: `7 passed`, the browser-contract assertions pass, and exit code is `0`.

- [ ] **Step 7: Perform the smallest browser check**

Generate a report with the synthetic payload used by the render test, open it by double-click, and select a small CSV containing quoted commas and line breaks. Confirm:

- headers populate the two column selectors;
- label scanning populates explicit positive/negative selectors;
- real progress advances during both passes;
- analysis values match the small file;
- disabling motion at operating-system level leaves all content readable;
- Developer Tools Network shows no requests.

- [ ] **Step 8: Review the Task 5 diff and request commit approval**

Run:

```powershell
git diff -- src\report_template.html src\vendor\papaparse.min.js tests\test_pipeline.py
```

After explicit approval only:

```powershell
git add src\report_template.html src\vendor\papaparse.min.js tests\test_pipeline.py
git commit -m "feat: analyze local CSVs in the report"
```

---

### Task 6: End-to-End CLI, IMDb Run, and Interview Deliverable

**Files:**
- Modify: `src/build_report.py`
- Modify: `src/report_template.html`
- Modify: `tests/test_pipeline.py`
- Create: `reports/imdb_analysis.html`

**Interfaces:**
- Produces: `build_analysis(input_path: Path, text_column: str, label_column: str, positive_label: str, negative_label: str) -> dict[str, object]`
- Produces: `main(argv: list[str] | None = None) -> int`
- CLI flags: `--input`, `--text-column`, `--label-column`, `--positive-label`, `--negative-label`, `--output`

- [ ] **Step 1: Append a failing end-to-end CLI test**

Append:

```python
from src.build_report import main


def test_cli_generates_report_from_balanced_csv(tmp_path: Path) -> None:
    rows = ["review,sentiment"]
    rows.extend(f'"excellent warm story {index}",positive' for index in range(20))
    rows.extend(f'"awful cold story {index}",negative' for index in range(20))
    source = write_csv(tmp_path, "\n".join(rows) + "\n")
    template = Path("src/report_template.html")
    papa = Path("src/vendor/papaparse.min.js")
    assert template.is_file() and papa.is_file()
    output = tmp_path / "analysis.html"

    exit_code = main([
        "--input", str(source),
        "--text-column", "review",
        "--label-column", "sentiment",
        "--positive-label", "positive",
        "--negative-label", "negative",
        "--output", str(output),
    ])

    assert exit_code == 0
    rendered = output.read_text(encoding="utf-8")
    assert "Naive Bayes" in rendered
    assert "Regressão Logística" in rendered
    assert '"train"' in rendered
    assert '"validation"' in rendered
    assert '"test"' in rendered
```

- [ ] **Step 2: Run the CLI test to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py::test_cli_generates_report_from_balanced_csv -v
```

Expected: collection fails because `main` does not exist.

- [ ] **Step 3: Implement payload assembly and CLI orchestration**

Add imports:

```python
import argparse
import platform
from datetime import datetime, timezone
from importlib.metadata import version
```

Add `build_analysis` so it calls, in order, `load_csv`, `encode_rows`, `profile_rows`, `deduplicate_rows`, `split_rows`, and `run_models`. Return only JSON-serializable values under these exact top-level keys:

```python
{
    "meta": {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_name": input_path.name,
        "python": platform.python_version(),
        "scikit_learn": version("scikit-learn"),
    },
    "quality": {**load_stats, **quality_stats, **duplicate_stats},
    "eda": profile,
    "splits": {name: len(rows) for name, rows in splits.items()},
    "modeling": model_results,
    "decisions": [
        "Bag of Words mantém o baseline simples e interpretável.",
        "Negações são preservadas porque alteram o sentimento.",
        "Stemming e lematização foram omitidos por serem opcionais.",
        "Duplicatas foram removidas antes do split para evitar vazamento.",
        "O teste foi consultado uma única vez após congelar as decisões.",
    ],
}
```

Implement `main` with `argparse`, defaults for the current IMDb columns and labels, paths relative to the repository root for the template and Papa Parse source, and a concise success line containing output path, unique-row count, selected model, and test F1 values. Catch `ValueError` at the CLI boundary, print its message to stderr, and return `2`; let unexpected exceptions fail with a traceback.

- [ ] **Step 4: Populate every report section from the payload**

Use text content and DOM construction rather than assigning untrusted values to `innerHTML`. Include:

- objective and dataset description;
- actual EDA counts and distributions;
- before/after deduplication counts;
- split sizes and leakage explanation;
- validation and test metrics for both models;
- five-fold F1 values, mean, and standard deviation for the selected candidate;
- confusion matrices with explicit actual/predicted axes;
- observed false-positive and false-negative excerpts;
- technical decisions, limitations, conclusions, and actionable next steps;
- a visible note that browser-selected CSV data stays local and receives no model training.

Add fixed containers such as `#dataset-size`, `#class-balance`, `#split-sizes`, `#eda-samples`, `#model-comparison`, `#cross-validation`, `#confusion-matrices`, and `#error-examples` to their matching sections. Populate them through helpers that use `textContent`:

```javascript
function setText(id, value) {
  document.getElementById(id).textContent = String(value);
}

function appendMetric(container, label, value) {
  const card = document.createElement("article");
  card.className = "card";
  const heading = document.createElement("h3");
  const metric = document.createElement("p");
  heading.textContent = label;
  metric.className = "metric";
  metric.textContent = typeof value === "number" ? value.toLocaleString("pt-BR") : value;
  card.append(heading, metric);
  container.append(card);
}

function renderStoredReport(report) {
  const modelLabels = {naive_bayes:"Naive Bayes", logistic_regression:"Regressão Logística"};
  setText("dataset-size", report.eda.rows);
  setText("class-balance",
    `${report.eda.class_counts.positive} positivas · ${report.eda.class_counts.negative} negativas`);
  setText("split-sizes",
    `${report.splits.train} treino · ${report.splits.validation} validação · ${report.splits.test} teste`);

  const edaSamples = document.getElementById("eda-samples");
  for (const [label, samples] of Object.entries(report.eda.samples)) {
    for (const sample of samples) appendMetric(edaSamples, `Amostra ${label}`, sample.slice(0, 500));
  }

  const comparison = document.getElementById("model-comparison");
  for (const [name, result] of Object.entries(report.modeling.models)) {
    for (const split of ["validation", "test"]) {
      for (const [metric, value] of Object.entries(result[split].metrics)) {
        if (metric !== "confusion_matrix") {
          appendMetric(comparison, `${modelLabels[name]} · ${split} · ${metric}`, value);
        }
      }
    }
  }
  setText("cross-validation",
    `${report.modeling.cross_validation_f1.join(" · ")} · média ${report.modeling.cross_validation_mean} · desvio ${report.modeling.cross_validation_std}`);

  const matrices = document.getElementById("confusion-matrices");
  const errors = document.getElementById("error-examples");
  for (const [name, result] of Object.entries(report.modeling.models)) {
    appendMetric(matrices, `${modelLabels[name]} · TN/FP/FN/TP`, result.test.metrics.confusion_matrix.flat().join(" / "));
    for (const [kind, samples] of Object.entries(result.test.errors)) {
      for (const sample of samples) appendMetric(errors, `${modelLabels[name]} · ${kind}`, sample.text);
    }
  }

  const decisionList = document.getElementById("decision-list");
  decisionList.replaceChildren(...report.decisions.map(decision => {
    const item = document.createElement("li");
    item.textContent = decision;
    return item;
  }));
}
renderStoredReport(REPORT);
```

Label the confusion-matrix cells in the surrounding static HTML as actual negative, actual positive, predicted negative, and predicted positive; do not rely only on the compact TN/FP/FN/TP summary used by the helper above.

- [ ] **Step 5: Run the complete synthetic suite to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_pipeline.py -v
```

Expected: `8 passed` and exit code `0`.

- [ ] **Step 6: Record the raw-dataset hash before the full run**

Run:

```powershell
$imdbBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath "IMDB%20Dataset.csv").Hash
$imdbBefore
```

Expected: one 64-character SHA-256 value. Retain `$imdbBefore` in the same PowerShell session through Step 8.

- [ ] **Step 7: Generate the real interview report**

Run:

```powershell
.\.venv\Scripts\python.exe src\build_report.py `
  --input "IMDB%20Dataset.csv" `
  --text-column review `
  --label-column sentiment `
  --positive-label positive `
  --negative-label negative `
  --output "reports\imdb_analysis.html"
```

Expected: exit code `0`; console output reports 50,000 source rows, 49,582 unique rows, the selected model, and final metrics for both models. `reports\imdb_analysis.html` exists and has non-zero length.

- [ ] **Step 8: Prove the raw dataset was not modified**

Run in the same PowerShell session as Step 6:

```powershell
$imdbAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath "IMDB%20Dataset.csv").Hash
if ($imdbBefore -ne $imdbAfter) { throw "O CSV original foi alterado" }
```

Expected: exit code `0` and no exception.

- [ ] **Step 9: Verify the generated artifact and Git scope**

Run:

```powershell
Get-Item reports\imdb_analysis.html | Select-Object FullName,Length,LastWriteTime
Select-String -Path reports\imdb_analysis.html -Pattern 'src="http','href="http','fetch\(','XMLHttpRequest'
git status --short
```

Expected: the report exists; `Select-String` returns no matches; Git status includes source, test, requirements, documentation, vendor, and report changes but excludes `IMDB%20Dataset.csv` and `.venv`.

- [ ] **Step 10: Perform final visual and interaction QA**

Open `reports/imdb_analysis.html` directly from File Explorer and verify:

- every required section is readable at desktop and narrow widths;
- metric values match the console payload;
- pipeline stages and bars animate from actual values;
- confusion-matrix labels are unambiguous;
- error examples contain observed texts and safe truncation;
- the local CSV workflow succeeds with `IMDB%20Dataset.csv` after explicit selections;
- invalid/equal label choices display a message and no misleading result;
- Developer Tools Network shows no request;
- reduced-motion mode preserves all content.

- [ ] **Step 11: Review the complete implementation and request commit approval**

Run:

```powershell
git diff --stat
git diff --check
.\.venv\Scripts\python.exe -m pytest -q
git status --short
```

Expected: `git diff --check` exits `0`; the full test suite exits `0`; only intended project artifacts appear in status.

After explicit approval only:

```powershell
git add .gitignore requirements.txt src tests reports docs\superpowers
git commit -m "feat: deliver portable IMDb sentiment analysis"
```

Do not push, publish, host, or create a pull request without separate explicit authorization.
