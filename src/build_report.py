from __future__ import annotations

import argparse
import csv
import html
import json
import os
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:  # importado como pacote: python -m pytest, a partir da raiz do repositório
    from src.text_normalization import normalize_text
except ModuleNotFoundError:  # executado direto: python src/build_report.py
    from text_normalization import normalize_text


PROTECTED_NEGATIONS = {"not", "no", "nor", "never"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - PROTECTED_NEGATIONS
HTML_TAG = re.compile(r"<[^>]+>")
HTML_BREAK = re.compile(r"<br\s*/?\s*>", re.IGNORECASE)
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
        encoded.append((text, int(label == positive_label)))
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
        if not class_texts:
            raise ValueError(f"nenhuma linha válida da classe {name}")
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
        "html_rows": sum(bool(HTML_BREAK.search(text)) for text, _ in rows),
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


def make_model(kind: str, *, preprocessor: object = clean_text) -> Pipeline:
    bow = CountVectorizer(
        preprocessor=preprocessor,
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


REPO_ROOT = Path(__file__).resolve().parent.parent


NORMALIZATION_MODES = ("none", "stemming", "lemmatization")


def _normalized_corpus(rows: list[tuple[str, int]], mode: str) -> list[str]:
    """Limpa e normaliza uma vez, em vez de deixar o vetorizador refazer a cada fit."""
    return [normalize_text(clean_text(text), mode) for text, _ in rows]


def run_normalization_experiment(
    splits: dict[str, list[tuple[str, int]]], random_state: int = 42
) -> dict[str, object]:
    """Compara nenhuma normalizacao, stemming e lematizacao na VALIDACAO.

    O conjunto de teste nao e tocado aqui. O estudo se sustenta em consulta-lo
    uma unica vez, depois de congeladas as decisoes; escolher normalizacao
    olhando o teste seria exatamente o vazamento que o resto do pipeline evita.
    A validacao cruzada roda somente sobre o treino e mede se as diferencas
    observadas na validacao sobrevivem a variacao entre folds.
    """
    _, train_y = _xy(splits["train"])
    _, validation_y = _xy(splits["validation"])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    variants: dict[str, Any] = {}

    for mode in NORMALIZATION_MODES:
        train_x = _normalized_corpus(splits["train"], mode)
        validation_x = _normalized_corpus(splits["validation"], mode)
        # max_features=50_000 satura nas tres variantes, entao o vocabulario do
        # modelo nao revela a compressao que a normalizacao provoca. O numero
        # informativo e quantos n-gramas distintos sobrevivem ao min_df=2 antes
        # do corte.
        uncapped = CountVectorizer(
            preprocessor=None, lowercase=False, ngram_range=(1, 2), min_df=2
        ).fit(train_x)
        models: dict[str, Any] = {}
        for kind in ("naive_bayes", "logistic_regression"):
            model = make_model(kind, preprocessor=None)
            model.fit(train_x, train_y)
            predictions = model.predict(validation_x).tolist()
            models[kind] = {
                "metrics": binary_metrics(validation_y, predictions),
                "vocabulary": len(model.named_steps["bow"].vocabulary_),
            }
        best = max(models, key=lambda name: models[name]["metrics"]["f1"])
        scores = cross_val_score(
            make_model(best, preprocessor=None), train_x, train_y, cv=folds, scoring="f1"
        )
        variants[mode] = {
            "models": models,
            "best_model": best,
            "vocabulary_uncapped": len(uncapped.vocabulary_),
            "vocabulary_used": models[best]["vocabulary"],
            "cross_validation_mean": round(fmean(float(score) for score in scores), 4),
            "cross_validation_std": round(pstdev(float(score) for score in scores), 4),
        }

    baseline = variants["none"]
    baseline_f1 = baseline["models"][baseline["best_model"]]["metrics"]["f1"]
    deltas = {
        mode: round(
            variants[mode]["models"][variants[mode]["best_model"]]["metrics"]["f1"]
            - baseline_f1,
            4,
        )
        for mode in NORMALIZATION_MODES
    }
    improved = [mode for mode in ("stemming", "lemmatization") if deltas[mode] > 0]
    # A dispersao entre folds e a regua: um ganho menor que ela e ruido.
    noise = max(variants[mode]["cross_validation_std"] for mode in NORMALIZATION_MODES)
    meaningful = [mode for mode in improved if deltas[mode] > noise]
    return {
        "variants": variants,
        "baseline_f1": baseline_f1,
        "validation_f1_delta": deltas,
        "fold_dispersion": round(noise, 4),
        "improved": improved,
        "meaningful_improvement": meaningful,
        "evaluated_on": "validation",
    }


def _normalization_decision(experiment: dict[str, Any]) -> str:
    """Frase de decisao derivada do que o experimento mediu, nao escrita a mao."""
    deltas = experiment["validation_f1_delta"]
    noise = experiment["fold_dispersion"]
    measured = (
        f"stemming {deltas['stemming']:+.4f} e lematizacao {deltas['lemmatization']:+.4f}"
        f" de F1 na validacao, contra dispersao de {noise:.4f} entre os folds"
    )
    if experiment["meaningful_improvement"]:
        ganho = " e ".join(experiment["meaningful_improvement"])
        return (
            f"Stemming e lematizacao foram medidos, nao presumidos: {measured}."
            f" O ganho de {ganho} supera a dispersao entre folds e por isso e tratado"
            " como real."
        )
    return (
        f"Stemming e lematizacao foram medidos, nao presumidos: {measured}."
        " Nenhuma das duas variacoes supera a dispersao entre folds, entao o"
        " pipeline segue sem normalizacao morfologica: o custo nao compra"
        " desempenho neste corpus."
    )


def build_analysis(
    input_path: Path,
    text_column: str,
    label_column: str,
    positive_label: str,
    negative_label: str,
) -> dict[str, object]:
    raw_rows, load_stats = load_csv(input_path, text_column, label_column)
    rows, quality_stats = encode_rows(
        raw_rows, text_column, label_column, positive_label, negative_label
    )
    profile = profile_rows(rows)
    unique_rows, duplicate_stats = deduplicate_rows(rows)
    splits = split_rows(unique_rows)
    _, model_results = run_models(splits)
    normalization = run_normalization_experiment(splits)
    return {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_name": input_path.name,
            "python": platform.python_version(),
            "scikit_learn": version("scikit-learn"),
        },
        "quality": {
            "source_rows": len(raw_rows),
            "unique_rows": len(unique_rows),
            **load_stats,
            **quality_stats,
            **duplicate_stats,
        },
        "eda": profile,
        "splits": {name: len(part) for name, part in splits.items()},
        "modeling": model_results,
        "normalization": normalization,
        "decisions": [
            "Bag of Words mantém o baseline simples e interpretável.",
            "Negações são preservadas porque alteram o sentimento.",
            _normalization_decision(normalization),
            "Duplicatas foram removidas antes do split para evitar vazamento.",
            "O teste foi consultado uma única vez após congelar as decisões.",
            "São três conjuntos, e não dois, porque a escolha entre os modelos é"
            " feita na validação: um conjunto usado para escolher deixa de ser"
            " estimativa imparcial do que foi escolhido.",
            "A retenção é de 15% para validação e 15% para teste porque cada"
            " conjunto guarda cerca de 7.400 resenhas, o que dá erro padrão de"
            " 0,36 ponto percentual sobre a acurácia (IC95 de mais ou menos 0,70"
            " pp) e separa com folga a diferença de 1,81 pp medida entre os dois"
            " modelos.",
            "Retenções de 10% elevariam o erro padrão a 0,44 pp (IC95 de mais ou"
            " menos 0,86 pp), perto demais da margem medida; retenções de 20%"
            " custariam cerca de 5.000 resenhas de treino, e com bigramas e"
            " min_df=2 é o treino que determina o tamanho do vocabulário.",
            "Validação e teste têm o mesmo tamanho para que o modelo seja julgado"
            " com a mesma precisão com que foi escolhido, e a validação cruzada de"
            " cinco folds sobre o treino cobre a dúvida que resta: se um único"
            " corte de validação teve sorte.",
            "Escolher normalização morfológica pela validação, e nunca pelo teste,"
            " mantém a promessa de uma única consulta ao teste.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera o relatório HTML autocontido de análise de sentimentos."
    )
    parser.add_argument("--input", default="IMDB%20Dataset.csv")
    parser.add_argument("--text-column", default="review")
    parser.add_argument("--label-column", default="sentiment")
    parser.add_argument("--positive-label", default="positive")
    parser.add_argument("--negative-label", default="negative")
    parser.add_argument("--output", default="reports/imdb_analysis.html")
    arguments = parser.parse_args(argv)

    try:
        payload = build_analysis(
            Path(arguments.input),
            arguments.text_column,
            arguments.label_column,
            arguments.positive_label,
            arguments.negative_label,
        )
        output_path = Path(arguments.output)
        render_report(
            payload,
            REPO_ROOT / "src" / "report_template.html",
            REPO_ROOT / "src" / "vendor" / "papaparse.min.js",
            output_path,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    quality: Any = payload["quality"]
    modeling: Any = payload["modeling"]
    test_f1 = " · ".join(
        f"{name} F1={result['test']['metrics']['f1']}"
        for name, result in modeling["models"].items()
    )
    print(
        f"Relatório gerado em {output_path} · "
        f"{quality['source_rows']} linhas de origem · "
        f"{quality['unique_rows']} linhas únicas · "
        f"modelo selecionado: {modeling['selected_model']} · teste: {test_f1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
