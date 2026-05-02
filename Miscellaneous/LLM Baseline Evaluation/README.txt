This module evaluates an LLM on predefined scenarios and compares predictions against ground truth decisions.
1.  input/Scenario.txt
    Test scenarios (inputs + expected conditions)
2.  test_LLM_Baseline.py
    Main script — runs LLM, generates predictions, computes metrics
3.  output/baseline_results.csv
    Raw predictions vs ground truth
4.  output/baseline_metrics.csv
    Overall performance (accuracy, precision, recall, F1)
5.  output/confusion_matrix.csv
    Breakdown of prediction errors across decision classes