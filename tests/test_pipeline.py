import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src import text_normalization
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


from src.build_report import render_report


REQUIRED_SECTION_IDS = {
    "summary", "method", "pipeline", "eda", "leakage",
    "models", "normalization", "errors", "decisions", "conclusion"
}


def test_rendered_report_is_self_contained(tmp_path: Path) -> None:
    template = tmp_path / "template.html"
    template.write_text(
        '<!doctype html><html><body><script>const REPORT = __REPORT_DATA__;</script>'
        + "".join(f'<section id="{name}"></section>' for name in REQUIRED_SECTION_IDS)
        + "</body></html>",
        encoding="utf-8",
    )
    output = tmp_path / "report.html"

    render_report({"title": "IMDb </script> safe"}, template, output)
    rendered = output.read_text(encoding="utf-8")

    assert "__REPORT_DATA__" not in rendered
    assert "<\\/script>" in rendered
    assert 'src="http' not in rendered
    assert 'href="http' not in rendered
    assert all(f'id="{name}"' in rendered for name in REQUIRED_SECTION_IDS)


def test_real_template_excludes_local_csv_analyzer(tmp_path: Path) -> None:
    output = tmp_path / "real-report.html"
    render_report(
        {},
        Path("src/report_template.html"),
        output,
    )
    rendered = output.read_text(encoding="utf-8")

    assert "Analise outro CSV" not in rendered
    assert 'id="local-csv"' not in rendered
    assert "Papa.parse" not in rendered
    assert 'src="http' not in rendered
    assert 'href="http' not in rendered


from src.build_report import main, run_normalization_experiment


def test_cli_generates_report_from_balanced_csv(tmp_path: Path) -> None:
    rows = ["review,sentiment"]
    rows.extend(f'"excellent warm story {index}",positive' for index in range(20))
    rows.extend(f'"awful cold story {index}",negative' for index in range(20))
    source = write_csv(tmp_path, "\n".join(rows) + "\n")
    template = Path("src/report_template.html")
    assert template.is_file()
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


def test_encode_rows_stores_stripped_text_so_padding_cannot_split_a_group(
    tmp_path: Path,
) -> None:
    path = write_csv(
        tmp_path,
        'review,sentiment\n"  padded review  ",positive\n"padded review",positive\n',
    )

    raw_rows, _ = load_csv(path, "review", "sentiment")
    rows, _ = encode_rows(raw_rows, "review", "sentiment", "positive", "negative")

    assert rows == [("padded review", 1), ("padded review", 1)]
    unique_rows, duplicate_stats = deduplicate_rows(rows)
    assert unique_rows == [("padded review", 1)]
    assert duplicate_stats == {"duplicate_groups": 1, "duplicate_rows": 1}


def test_profile_rows_rejects_a_single_class_corpus() -> None:
    with pytest.raises(ValueError, match="nenhuma linha válida da classe positive"):
        profile_rows([("only negative text", 0)])


def test_generated_report_renders_under_a_stubbed_dom(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node não está disponível para o teste de renderização")
    rows = ["review,sentiment"]
    rows.extend(f'"excellent warm story {index}",positive' for index in range(20))
    rows.extend(f'"awful cold story {index}",negative' for index in range(20))
    source = write_csv(tmp_path, "\n".join(rows) + "\n")
    output = tmp_path / "smoke.html"

    assert main([
        "--input", str(source),
        "--text-column", "review",
        "--label-column", "sentiment",
        "--positive-label", "positive",
        "--negative-label", "negative",
        "--output", str(output),
    ]) == 0

    result = subprocess.run(
        [node, "tests/report_smoke.js", str(output)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_normalization_experiment_never_touches_the_test_split() -> None:
    train = [
        *((f"excellent warm stories {index}", 1) for index in range(12)),
        *((f"awful cold stories {index}", 0) for index in range(12)),
    ]
    validation = [
        *((f"excellent warm validating {index}", 1) for index in range(5)),
        *((f"awful cold validating {index}", 0) for index in range(5)),
    ]
    marker = "testonlyword"
    test = [
        *((f"excellent warm {marker} {index}", 1) for index in range(5)),
        *((f"awful cold {marker} {index}", 0) for index in range(5)),
    ]

    experiment = run_normalization_experiment(
        {"train": train, "validation": validation, "test": test}
    )

    assert experiment["evaluated_on"] == "validation"
    assert set(experiment["variants"]) == {"none", "stemming", "lemmatization"}
    # O marcador exclusivo do teste não pode ter alcançado nenhuma variante: o
    # experimento escolhe normalização sem consultar o conjunto de teste.
    assert marker not in json.dumps(experiment)
    for variant in experiment["variants"].values():
        assert variant["vocabulary_uncapped"] >= variant["vocabulary_used"]
        assert set(variant["models"]) == {"naive_bayes", "logistic_regression"}
    assert experiment["validation_f1_delta"]["none"] == 0.0


def test_text_normalization_self_check_passes() -> None:
    # O módulo carrega as asserções do stemmer e do lematizador em demo();
    # aqui elas entram na suíte em vez de dependerem de execução manual.
    text_normalization.demo()
