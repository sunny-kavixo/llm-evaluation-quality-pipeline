-- Example analytics after loading evaluation_results into llm_evaluations
SELECT ROUND(AVG(overall),3) AS avg_overall,
       ROUND(AVG(relevance),3) AS avg_relevance,
       ROUND(AVG(hallucination_risk),3) AS avg_hallucination_risk
FROM llm_evaluations;

SELECT CASE WHEN overall >= .80 THEN 'high' WHEN overall >= .60 THEN 'medium' ELSE 'low' END AS quality_band,
       COUNT(*) AS responses
FROM llm_evaluations
GROUP BY 1 ORDER BY 1;
