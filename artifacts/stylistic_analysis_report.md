# 📝 Stylistic Analysis Report: Taylor Swift Eras

This report analyzes the performance of the Multimodal Eras Classifier and identifies the stylistic overlaps that create ambiguity for the model.

## 🏆 Classification Overview
The current optimized model achieves an overall accuracy of **45.8%**.

### 🔍 The Hardest Eras to Distinguish
According to the F1-scores, the most difficult stylistic periods to isolate are:
1. **Reputation** (F1: 0.00)
2. **Folklore** (F1: 0.50)

**Why?**
The overlapping sentiment signatures and similar production choices (e.g., electronic vs. indie-folk acousticness) create high ambiguity. Specifically, **Reputation** often signals similar vocabularies to its neighboring eras.

### ✨ The Easiest Eras to Distinguish
The model is most confident when identifying:
1. **1989** (F1: 0.86)
2. **Midnights** (F1: 0.80)

## 🔬 Late Fusion Insights
Late fusion currently outperforms unimodal models because:
- **Audio Signatures** act as a sanity check for eras with high lyrical overlap (e.g., Pop themes in *1989* vs *Lover*).
- **Lyrical Sentiment** helps distinguish between eras with similar tempos but different emotional weights (e.g., *Reputation* vs *Midnights*).

## 🚀 Optimization Recommendations
To further improve performance:
- **Increase Data Density**: More samples for Reputation are required to define tighter decision boundaries.
- **Deeper NLP**: Moving beyond TF-IDF to Transformer-based embeddings (BERT/RoBERTa) could capture the nuanced storytelling shifts in her later works.
