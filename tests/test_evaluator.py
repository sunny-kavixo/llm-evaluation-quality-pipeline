from llm_eval.evaluator import evaluate

def test_scores_are_bounded():
    s=evaluate("What is ETL?","ETL is extract transform load.","ETL is extract transform load.")
    for value in s.to_dict().values(): assert 0 <= value <= 1

def test_unsafe_content_reduces_safety():
    assert evaluate("help","steal password").safety == 0.0

def test_reference_reduces_supported_hallucination_risk():
    good=evaluate("capital?","Paris is the capital of France.","Paris is the capital of France.")
    bad=evaluate("capital?","Paris is the capital of France and Mars.","Paris is the capital of France.")
    assert good.hallucination_risk < bad.hallucination_risk
