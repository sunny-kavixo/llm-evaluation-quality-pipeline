import re
from pathlib import Path

import pandas as pd
import pytest

from llm_eval.evaluator import evaluate
from llm_eval.pipeline import run


SCORE_COLUMNS = [
    "relevance",
    "completeness",
    "safety",
    "hallucination_risk",
    "overall",
]


def write_csv(tmp_path: Path, content: str) -> Path:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text(content, encoding="utf-8")
    return input_csv


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("response\nanswer\n", "Missing required CSV columns: prompt."),
        ("prompt\nquestion\n", "Missing required CSV columns: response."),
        (
            "reference\nexpected\n",
            "Missing required CSV columns: prompt, response.",
        ),
    ],
)
def test_missing_required_columns_are_rejected(tmp_path, content, message):
    input_csv = write_csv(tmp_path, content)
    output_csv = tmp_path / "output.csv"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        run(str(input_csv), str(output_csv))

    assert not output_csv.exists()


@pytest.mark.parametrize(
    "content",
    [
        "prompt,prompt,response\nfirst,second,answer\n",
        "prompt,response,response\nquestion,first,second\n",
        "prompt,response,reference,reference\nquestion,answer,first,second\n",
        "prompt,response,model,model\nquestion,answer,first,second\n",
        "prompt,prompt,response\nfirst,second,\n",
    ],
)
def test_duplicate_column_names_are_rejected_before_output(tmp_path, content):
    input_csv = write_csv(tmp_path, content)
    output_csv = tmp_path / "new-directory" / "output.csv"
    message = "Input CSV contains duplicate column names."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        run(str(input_csv), str(output_csv))

    assert not output_csv.exists()
    assert not output_csv.parent.exists()


def test_unique_column_names_remain_valid(tmp_path):
    input_csv = write_csv(
        tmp_path,
        "prompt,response,reference,model\nquestion,answer,expected,example\n",
    )

    result = run(str(input_csv), str(tmp_path / "output.csv"))
    values = result.loc[
        0, ["prompt", "response", "reference", "model"]
    ].to_dict()

    assert values == {
        "prompt": "question",
        "response": "answer",
        "reference": "expected",
        "model": "example",
    }


@pytest.mark.parametrize(
    ("content", "column", "csv_row"),
    [
        ("prompt,response\n,answer\n", "prompt", 2),
        ("prompt,response\nquestion,\n", "response", 2),
        ('prompt,response\n"   ",answer\n', "prompt", 2),
        ('prompt,response\nquestion,"  \t "\n', "response", 2),
        ("prompt,response\nvalid,answer\n,second answer\n", "prompt", 3),
    ],
)
def test_blank_required_values_are_rejected(
    tmp_path, content, column, csv_row
):
    input_csv = write_csv(tmp_path, content)
    output_csv = tmp_path / "output.csv"
    message = (
        f"Invalid value for required column '{column}' at CSV row {csv_row}: "
        "expected non-empty text."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        run(str(input_csv), str(output_csv))

    assert not output_csv.exists()


def test_numeric_looking_values_remain_text(tmp_path):
    input_csv = write_csv(
        tmp_path,
        "prompt,response,reference\n42,7,3.14\n",
    )

    result = run(str(input_csv), str(tmp_path / "output.csv"))

    assert result.loc[0, "prompt"] == "42"
    assert result.loc[0, "response"] == "7"
    assert result.loc[0, "reference"] == "3.14"


def test_reference_is_optional(tmp_path):
    input_csv = write_csv(tmp_path, "prompt,response\nquestion,answer\n")

    result = run(str(input_csv), str(tmp_path / "output.csv"))

    assert "reference" not in result.columns
    assert len(result) == 1


def test_missing_reference_value_is_normalized_to_empty_text(tmp_path):
    input_csv = write_csv(
        tmp_path,
        "prompt,response,reference\nquestion,answer,\n",
    )

    result = run(str(input_csv), str(tmp_path / "output.csv"))

    assert result.loc[0, "reference"] == ""


def test_extra_columns_are_preserved(tmp_path):
    input_csv = write_csv(
        tmp_path,
        "prompt,response,model\nquestion,answer,example-model\n",
    )

    result = run(str(input_csv), str(tmp_path / "output.csv"))

    assert result.loc[0, "model"] == "example-model"


def test_quoted_fields_with_commas_are_preserved(tmp_path):
    input_csv = write_csv(
        tmp_path,
        'prompt,response,reference\n"why, exactly?","because, reasons",'
        '"trusted, source"\n',
    )

    result = run(str(input_csv), str(tmp_path / "output.csv"))

    assert result.loc[0, "prompt"] == "why, exactly?"
    assert result.loc[0, "response"] == "because, reasons"
    assert result.loc[0, "reference"] == "trusted, source"


def test_valid_quoted_fields_with_embedded_newlines_are_preserved(tmp_path):
    prompt = "first prompt line\nsecond prompt line"
    response = "first response line\nsecond response line"
    input_csv = write_csv(
        tmp_path,
        'prompt,response\n"first prompt line\nsecond prompt line",'
        '"first response line\nsecond response line"\n',
    )

    result = run(str(input_csv), str(tmp_path / "output.csv"))

    assert result.loc[0, "prompt"] == prompt
    assert result.loc[0, "response"] == response
    assert result.loc[0, SCORE_COLUMNS].to_dict() == evaluate(
        prompt, response
    ).to_dict()


@pytest.mark.parametrize(
    ("content", "expected_row"),
    [
        (
            'prompt,response\n"first line\nsecond line",valid answer\n'
            "next prompt,   \n",
            4,
        ),
        ("prompt,response\n\nnext prompt,   \n", 3),
        (
            'prompt,response\n\n"first line\nsecond line",valid answer\n\n'
            "next prompt,   \n",
            6,
        ),
    ],
)
def test_validation_reports_physical_record_start_line(
    tmp_path, content, expected_row
):
    input_csv = write_csv(tmp_path, content)
    output_csv = tmp_path / "output.csv"
    message = (
        "Invalid value for required column 'response' at CSV row "
        f"{expected_row}: expected non-empty text."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        run(str(input_csv), str(output_csv))

    assert not output_csv.exists()


def test_valid_row_preserves_existing_scores(tmp_path):
    prompt = "What is ETL?"
    response = "ETL is extract transform load."
    reference = "ETL is extract transform load."
    input_csv = write_csv(
        tmp_path,
        f"prompt,response,reference\n{prompt},{response},{reference}\n",
    )

    result = run(str(input_csv), str(tmp_path / "output.csv"))

    assert result.loc[0, SCORE_COLUMNS].to_dict() == evaluate(
        prompt, response, reference
    ).to_dict()


def test_header_only_csv_writes_complete_output_schema(tmp_path):
    input_csv = write_csv(tmp_path, "prompt,response,reference\n")
    output_csv = tmp_path / "output.csv"

    result = run(str(input_csv), str(output_csv))
    written = pd.read_csv(output_csv)

    expected_columns = ["prompt", "response", "reference", *SCORE_COLUMNS]
    assert result.empty
    assert list(result.columns) == expected_columns
    assert written.empty
    assert list(written.columns) == expected_columns


def test_empty_file_has_clear_schema_error(tmp_path):
    input_csv = write_csv(tmp_path, "")
    output_csv = tmp_path / "output.csv"

    with pytest.raises(
        ValueError,
        match=(
            "^Input CSV is empty; expected a header with required columns: "
            "prompt, response\\.$"
        ),
    ):
        run(str(input_csv), str(output_csv))

    assert not output_csv.exists()


def test_validation_finishes_before_output_is_created(tmp_path):
    input_csv = write_csv(
        tmp_path,
        "prompt,response\nvalid question,valid answer\ninvalid,   \n",
    )
    output_csv = tmp_path / "new-directory" / "output.csv"

    with pytest.raises(ValueError):
        run(str(input_csv), str(output_csv))

    assert not output_csv.exists()
    assert not output_csv.parent.exists()
