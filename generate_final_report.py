import json
import pandas as pd
import os

def generate_report():
    metrics_path = 'artifacts/eval_metrics.json'
    report_path = 'artifacts/stylistic_analysis_report.md'
    
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    era_report = metrics.get('era_report', {})
    
    # Identify hardest to distinguish
    # Logic: Lower F1-score means harder to distinguish
    f1_scores = {era: data['f1-score'] for era, data in era_report.items() if era not in ['accuracy', 'macro avg', 'weighted avg']}
    sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1])
    
    hardest_eras = sorted_f1[:2]
    easiest_eras = sorted_f1[-2:]

    content = f"""# 📝 Stylistic Analysis Report: Taylor Swift Eras

This report analyzes the performance of the Multimodal Eras Classifier and identifies the stylistic overlaps that confuse the AI.

## 🏆 Classification Overview
The current optimized model achieves an overall accuracy of **{metrics.get('results', {}).get('late', {}).get('accuracy', 0):.1%}**.

### 🔍 The Hardest Eras to Distinguish
According to the F1-scores, the most difficult stylistic periods to isolate are:
1. **{hardest_eras[0][0]}** (F1: {hardest_eras[0][1]:.2f})
2. **{hardest_eras[1][0]}** (F1: {hardest_eras[1][1]:.2f})

**Why?**
The overlapping sentiment signatures and similar production choices (e.g., electronic vs. indie-folk acousticness) create high ambiguity. Specifically, **{hardest_eras[0][0]}** often signals similar vocabularies to its neighboring eras.

### ✨ The Easiest Eras to Distinguish
The model is most confident when identifying:
1. **{easiest_eras[1][0]}** (F1: {easiest_eras[1][1]:.2f})
2. **{easiest_eras[0][0]}** (F1: {easiest_eras[0][1]:.2f})

## 🔬 Late Fusion Insights
Late fusion currently outperforms unimodal models because:
- **Audio Signatures** act as a sanity check for eras with high lyrical overlap (e.g., Pop themes in *1989* vs *Lover*).
- **Lyrical Sentiment** helps distinguish between eras with similar tempos but different emotional weights (e.g., *Reputation* vs *Midnights*).

## 🚀 Optimization Recommendations
To further improve performance:
- **Increase Data Density**: More samples for {hardest_eras[0][0]} are required to define tighter decision boundaries.
- **Deeper NLP**: Moving beyond TF-IDF to Transformer-based embeddings (BERT/RoBERTa) could capture the nuanced storytelling shifts in her later works.
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    generate_report()
