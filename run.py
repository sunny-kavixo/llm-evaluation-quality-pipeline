import argparse
from llm_eval.pipeline import run

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="data/sample_responses.csv")
    p.add_argument("--output",default="reports/evaluation_results.csv")
    a=p.parse_args()
    df=run(a.input,a.output)
    print(df[["relevance","completeness","safety","hallucination_risk","overall"]].mean().round(3))
