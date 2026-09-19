from pathlib import Path
import pandas as pd
from .evaluator import evaluate

def run(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv).fillna("")
    rows=[]
    for row in df.to_dict("records"):
        score=evaluate(row["prompt"], row["response"], row.get("reference",""))
        rows.append({**row, **score.to_dict()})
    out=pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv,index=False)
    return out
