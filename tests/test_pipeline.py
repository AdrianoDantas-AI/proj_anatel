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
