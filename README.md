# LLM Evaluation Quality Pipeline

A reproducible Python pipeline for evaluating prompt/response datasets across **relevance, completeness, safety, hallucination risk, and overall quality**. Built as a portfolio project for AI data quality and model-evaluation workflows.

## Features

- Batch evaluation of CSV prompt/response datasets
- Reference-aware hallucination-risk heuristic
- Relevance, completeness and safety scoring
- Pandas-based report generation
- SQL queries for aggregate quality analysis
- Pytest unit tests and GitHub Actions CI
- Dockerized execution
- Synthetic sample data only; no proprietary model outputs or claimed production metrics

> The included evaluator is a transparent heuristic baseline for demonstrating evaluation workflow design. It is not a substitute for factuality verification, human review, or a production safety classifier.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
pytest -q
```

Results are written to `reports/evaluation_results.csv`.

## Input schema

| column | purpose |
|---|---|
| prompt | user/task prompt |
| response | model response to evaluate |
| reference | optional trusted reference answer |

## Architecture

`CSV dataset -> evaluator -> scored Pandas DataFrame -> CSV report -> SQL analytics`

## Scoring

Each row receives values from 0 to 1 for relevance, completeness, safety, hallucination risk and an overall weighted score. The implementation is deterministic so tests and CI are reproducible.

## Docker

```bash
docker build -t llm-eval-pipeline .
docker run --rm llm-eval-pipeline
```

## Project structure

```text
llm_eval/          evaluation engine and batch pipeline
data/              synthetic example dataset
sql/               analytics queries
tests/             automated tests
.github/workflows/ CI
run.py              CLI entry point
Dockerfile          container runtime
```
