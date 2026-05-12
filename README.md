# 🎶 Taylor Swift Eras: Multimodal Classification Project

This project implements a multimodal machine learning pipeline to classify Taylor Swift's musical eras based on acoustic signatures (Spotify technical metrics) and semantic lyrical weight (Genius lyrics and VADER sentiment).

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure
*   `app.py`: Streamlit web application (Main Entry Point).
*   `train.py`: Training pipeline (Pre-processing, Engineering, Evaluation).
*   `src/`: Core logic modules.
    *   `model.py`: Multimodal fusion logic (Late Fusion ensemble).
    *   `audio_subclassifier.py`: Acoustic feature engineering and SVM classifier.
    *   `preprocessing.py`: NLP cleaning and sentiment analysis.
    *   `topic_modeling.py`: LDA-based semantic clustering.
*   `data/`: Raw and processed datasets.
*   `artifacts/`: Model evaluation metrics, confusion matrices, and reports.

## 🧪 Running the Pipeline

### Step 1: Data Preparation & Training
Run the training script to engineer features and optimize the late fusion ensemble. This will generate `model.pkl` and update the evaluation artifacts.

```bash
python train.py
```

### Step 2: Launch the Web Portal
Use Streamlit to launch the interactive dashboard for manual era prediction and model interpretability.

```bash
streamlit run app.py
```

## 📊 Methodology: Late Fusion Ensemble
The project uses a **Late Fusion** architecture to reconcile the different structural natures of the data:
1.  **Semantic Path**: Processes lyrics via sublinear TF-IDF and sentiment intensity.
2.  **Acoustic Path**: Analyzes 12+ Spotify acoustic metrics (Danceability, Acousticness, etc.) using normalized feature distributions.
3.  **Meta-Classifier**: Aggregates probabilities from both sub-models using a Logistic Regression decision layer to produce the final Era prediction.

## 👥 Team
This project is structured for a collaborative environment with components owned by five specialized roles:
*   **Member 1**: Data Collection & Lifecycle.
*   **Member 2**: NLP & Semantic Analysis.
*   **Member 3**: Acoustic Feature Engineering.
*   **Member 4**: ML Architecture & Fusion Logic.
*   **Member 5**: Implementation, UI, and Evaluation.
