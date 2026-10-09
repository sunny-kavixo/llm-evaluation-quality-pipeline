import csv
from dataclasses import fields
from pathlib import Path

import pandas as pd

from .evaluator import Evaluation, evaluate


REQUIRED_COLUMNS = ("prompt", "response")
TEXT_COLUMNS = (*REQUIRED_COLUMNS, "reference")


def _read_and_validate(input_csv: str) -> pd.DataFrame:
    try:
        with Path(input_csv).open(encoding="utf-8-sig", newline="") as input_file:
            reader = csv.reader(input_file, strict=True)
            header = None
            records = []
            record_start_lines = []
            previous_end_line = 0

            for record in reader:
                record_start_line = previous_end_line + 1
                previous_end_line = reader.line_num
                if not record:
                    continue
                if header is None:
                    header = record
                    if len(header) != len(set(header)):
                        raise ValueError(
                            "Input CSV contains duplicate column names."
                        )
                    continue
                if len(record) > len(header):
                    raise ValueError(
                        f"CSV record at physical line {record_start_line} has "
                        "more fields than the header."
                    )
                records.append([*record, *([""] * (len(header) - len(record)))])
                record_start_lines.append(record_start_line)
    except csv.Error as exc:
        raise ValueError("Input contains invalid CSV syntax.") from exc

    if header is None:
        raise ValueError(
            "Input CSV is empty; expected a header with required columns: "
            "prompt, response."
        )

    df = pd.DataFrame(records, columns=header)
    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}.")

    required_values = df.loc[:, REQUIRED_COLUMNS]
    for csv_row, values in zip(
        record_start_lines,
        required_values.itertuples(index=False, name=None),
    ):
        for column, value in zip(REQUIRED_COLUMNS, values):
            if pd.isna(value) or not value.strip():
                raise ValueError(
                    f"Invalid value for required column '{column}' at CSV row "
                    f"{csv_row}: expected non-empty text."
                )

    return df.fillna("")


def run(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = _read_and_validate(input_csv)
    rows = []
    for row in df.to_dict("records"):
        score = evaluate(row["prompt"], row["response"], row.get("reference", ""))
        rows.append({**row, **score.to_dict()})

    output_columns = list(df.columns)
    for field in fields(Evaluation):
        if field.name not in output_columns:
            output_columns.append(field.name)
    out = pd.DataFrame(rows, columns=output_columns)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
