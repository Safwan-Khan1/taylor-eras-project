import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from PIL import Image
from src.model import load_model, MultimodalAudioLyricsModel
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Taylor Swift Eras Classifier", layout="wide")

def set_bg_hack(main_bg):
    if os.path.exists(main_bg):
        import base64
        with open(main_bg, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded_string});
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }}
            .stApp > header {{
                background-color: transparent;
            }}
            .block-container {{
                background-color: rgba(0, 0, 0, 0.85);
                border-radius: 1rem;
                padding: 2rem;
                margin-top: 2rem;
            }}
            /* Song Testing Lab custom styles */
            .metric-card {{
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 0.75rem;
                padding: 1rem;
                margin-bottom: 0.5rem;
            }}
            .correct-badge {{
                background: linear-gradient(135deg, #11998e, #38ef7d);
                color: black;
                padding: 0.4rem 1rem;
                border-radius: 2rem;
                font-weight: bold;
                display: inline-block;
            }}
            .wrong-badge {{
                background: linear-gradient(135deg, #f7971e, #ffd200);
                color: black;
                padding: 0.4rem 1rem;
                border-radius: 2rem;
                font-weight: bold;
                display: inline-block;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Call the background function
set_bg_hack('Taylor swift background.png')

st.title("🎶 Taylor Swift Eras: Multimodal Classification")
st.markdown("""
Welcome to the interactive web portal! This tool analyzes the stylistic periods of Taylor Swift using both **acoustic signatures** and **semantic lyrical weight**.
Enter a song's lyrics and audio features below to predict which Era it belongs to!
""")

@st.cache_resource
def get_model():
    return load_model('model.pkl')

@st.cache_data
def load_test_songs():
    """Load pre-extracted test songs from the dataset."""
    path = 'test_songs_data.json'
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_full_dataset():
    """Load the full dataset for feature importance analysis."""
    path = 'data/raw/taylor_eras_multimodal_dataset.csv'
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

model = get_model()
metrics_path = 'artifacts/eval_metrics.json'

AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_ms', 'key', 'mode'
]

FEATURE_LABELS = {
    'danceability':     '💃 Danceability',
    'energy':           '⚡ Energy',
    'loudness':         '🔊 Loudness (dB)',
    'speechiness':      '🗣️ Speechiness',
    'acousticness':     '🎸 Acousticness',
    'instrumentalness': '🎹 Instrumentalness',
    'liveness':         '🎤 Liveness',
    'valence':          '😊 Valence',
    'tempo':            '🥁 Tempo (BPM)',
    'duration_ms':      '⏱️ Duration (ms)',
    'key':              '🎵 Key',
    'mode':             '🎼 Mode',
}

KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ERA_COLORS = {
    '1989': '#89CFF0',
    'Reputation': '#2d2d2d',
    'Lover': '#FF9EBC',
    'Folklore': '#C9B99A',
    'Evermore': '#A0785A',
    'Midnights': '#1B1464',
}

def predict_with_audio(audio_data, text_data: dict):
    """
    Build feature vector, apply overrides, and predict.
    audio_data can be a dictionary (from dataset) or a list (from sliders).
    text_data should contain 'lyrics'.
    """
    if isinstance(audio_data, dict):
        # Using a song dictionary with optional overrides in text_data (repurposed dict)
        overrides = text_data.get('overrides', {})
        lyrics = audio_data['lyrics']
        
        danceability = overrides.get('danceability', audio_data['danceability'])
        energy = overrides.get('energy', audio_data['energy'])
        loudness = overrides.get('loudness', audio_data['loudness'])
        speechiness = overrides.get('speechiness', audio_data['speechiness'])
        acousticness = overrides.get('acousticness', audio_data['acousticness'])
        instrumentalness = overrides.get('instrumentalness', audio_data['instrumentalness'])
        liveness = overrides.get('liveness', audio_data['liveness'])
        valence = overrides.get('valence', audio_data['valence'])
        tempo = overrides.get('tempo', audio_data['tempo'])
        duration_ms = overrides.get('duration_ms', audio_data['duration_ms'])
        key_val = int(overrides.get('key', audio_data['key']))
        mode_val = int(overrides.get('mode', audio_data['mode']))
    else:
        # audio_data is a raw list/vector from manual input
        # text_data contains {'lyrics': ...}
        vec = audio_data
        lyrics = text_data['lyrics']
        X_audio = np.array([vec])
        X_text = [lyrics]
        
        # In this branch, vec is already the full vector, so we skip the builder
        pred = model.predict(X_audio, X_text)[0]
        probs = model.predict_proba(X_audio, X_text)[0]
        
        # Extract unimodal probabilities for interpretability (defensive checks)
        audio_probs = None
        if hasattr(model, 'audio_classifier') and hasattr(model, 'audio_scaler'):
            try:
                audio_probs = model.audio_classifier.predict_proba(model.audio_scaler.transform(X_audio))[0]
            except:
                pass
                
        text_probs = None
        if hasattr(model, 'text_classifier') and hasattr(model, 'text_vectorizer'):
            try:
                text_probs = model.text_classifier.predict_proba(model.text_vectorizer.transform(X_text).toarray())[0]
            except:
                pass
        
        # Fallback to uniform distribution if model is unimodal
        num_classes = len(model.classes_)
        uniform_prob = 1.0 / num_classes
        
        if audio_probs is None:
            audio_probs = np.full(num_classes, uniform_prob)
        if text_probs is None:
            text_probs = np.full(num_classes, uniform_prob)
            
        unimodal = {
            'audio': dict(zip(model.classes_, audio_probs)),
            'text': dict(zip(model.classes_, text_probs)),
            'audio_active': (hasattr(model, 'audio_classifier')),
            'text_active': (hasattr(model, 'text_classifier'))
        }
        return pred, dict(zip(model.classes_, probs)), unimodal

    # Feature engineering for dictionary input branch
    acoustic_energy_ratio = acousticness / (energy + 0.001)
    dance_valence_product = danceability * valence
    loudness_norm = (loudness + 60) / 60
    
    if tempo <= 90: tempo_bin = 0.0
    elif tempo <= 110: tempo_bin = 1.0
    elif tempo <= 130: tempo_bin = 2.0
    elif tempo <= 160: tempo_bin = 3.0
    else: tempo_bin = 4.0
    speech_acoustic_diff = speechiness - acousticness

    vec = [
        danceability, energy, loudness, speechiness, 
        acousticness, instrumentalness, liveness, valence, 
        tempo, duration_ms,
        acoustic_energy_ratio, dance_valence_product, loudness_norm,
        tempo_bin, speech_acoustic_diff
    ]
    for k in range(12):
        vec.append(1.0 if key_val == k else 0.0)
    for m in [0, 1]:
        vec.append(1.0 if mode_val == m else 0.0)

    X_audio = np.array([vec])
    X_text  = [lyrics]
    pred    = model.predict(X_audio, X_text)[0]
    probs   = model.predict_proba(X_audio, X_text)[0]
    
    # Extract unimodal probabilities for debate (defensive checks)
    audio_probs = None
    if hasattr(model, 'audio_classifier') and hasattr(model, 'audio_scaler'):
        try:
            audio_probs = model.audio_classifier.predict_proba(model.audio_scaler.transform(X_audio))[0]
        except:
            pass
            
    text_probs = None
    if hasattr(model, 'text_classifier') and hasattr(model, 'text_vectorizer'):
        try:
            text_probs = model.text_classifier.predict_proba(model.text_vectorizer.transform(X_text).toarray())[0]
        except:
            pass
    
    # Fallback to uniform distribution if model is unimodal
    num_classes = len(model.classes_)
    uniform_prob = 1.0 / num_classes
    
    if audio_probs is None:
        audio_probs = np.full(num_classes, uniform_prob)
    if text_probs is None:
        text_probs = np.full(num_classes, uniform_prob)
        
    unimodal = {
        'audio': dict(zip(model.classes_, audio_probs)),
        'text': dict(zip(model.classes_, text_probs)),
        'audio_active': (hasattr(model, 'audio_classifier')),
        'text_active': (hasattr(model, 'text_classifier'))
    }
    
    return pred, dict(zip(model.classes_, probs)), unimodal

def feature_sensitivity_analysis(song, overrides):
    """
    Vary each audio feature ±20% of its range and measure how much the
    top-class probability changes. Higher delta → that feature is more
    influential for this song's prediction.
    """
    RANGES = {
        'danceability': (0.0, 1.0),
        'energy': (0.0, 1.0),
        'loudness': (-60.0, 0.0),
        'speechiness': (0.0, 1.0),
        'acousticness': (0.0, 1.0),
        'instrumentalness': (0.0, 1.0),
        'liveness': (0.0, 1.0),
        'valence': (0.0, 1.0),
        'tempo': (50.0, 250.0),
        'duration_ms': (90000, 400000),
        'key': (0, 11),
        'mode': (0, 1),
    }
    base_pred, base_probs, _ = predict_with_audio(song, {'overrides': overrides})
    base_top = max(base_probs.values())

    deltas = {}
    for feat in AUDIO_FEATURES:
        lo, hi = RANGES[feat]
        step = (hi - lo) * 0.20
        orig = overrides.get(feat, song[feat])

        low_val  = max(lo, orig - step)
        high_val = min(hi, orig + step)

        _, p_low, _  = predict_with_audio(song, {'overrides': {**overrides, feat: low_val}})
        _, p_high, _ = predict_with_audio(song, {'overrides': {**overrides, feat: high_val}})

        # Sensitivity = max change in top-class probability when feature is nudged
        delta = max(
            abs(max(p_low.values())  - base_top),
            abs(max(p_high.values()) - base_top)
        )
        deltas[feat] = delta

    return deltas

if not os.path.exists('model.pkl') or not os.path.exists(metrics_path):
    st.warning("Model and metrics not found. Please run `python train.py` first to generate the model and artifacts.")
else:
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    st.sidebar.header("Model Performance Metrics")
    st.sidebar.subheader("Experimental Results (5-Fold CV)")
    res_df = pd.DataFrame(metrics.get('results', {})).T
    st.sidebar.dataframe(res_df.style.format("{:.1%}"))

    st.sidebar.markdown(f"**Vocab Size**: {metrics['vocab_size']} words")
    st.sidebar.markdown(f"**Sparsity**: {metrics['sparsity_perc']*100:.2f}%")

    st.sidebar.subheader("Latent Dirichlet Allocation (Topics)")
    for topic, words in metrics['topics'].items():
        st.sidebar.write(f"**{topic}:** {', '.join(words)}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab3 = st.tabs(["🔮 Predict Era", "📊 Project Analysis"])

    # ── Tab 1: Manual Prediction ───────────────────────────────────────────────
    with tab1:
        st.subheader("Input Song Data")
        col1, col2 = st.columns([1, 1])

        with col1:
            lyrics_input = st.text_area("Song Lyrics", height=250, placeholder="Paste song lyrics here...")

        with col2:
            st.markdown("### Audio Features (Spotify API scaling)")
            danceability      = st.slider("Danceability", 0.0, 1.0, 0.5)
            energy            = st.slider("Energy", 0.0, 1.0, 0.5)
            loudness          = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
            speechiness       = st.slider("Speechiness", 0.0, 1.0, 0.1)
            acousticness      = st.slider("Acousticness", 0.0, 1.0, 0.2)
            instrumentalness  = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
            liveness          = st.slider("Liveness", 0.0, 1.0, 0.1)
            valence           = st.slider("Valence", 0.0, 1.0, 0.5)
            tempo             = st.slider("Tempo (BPM)", 50.0, 250.0, 120.0)
            duration_ms       = st.number_input("Duration (ms)", min_value=0, value=210000)
            key               = st.selectbox("Key (Pitch Class)", options=list(range(12)),
                                             format_func=lambda x: KEY_NAMES[x])
            mode              = st.radio("Mode", options=[0, 1],
                                         format_func=lambda x: "Major" if x == 1 else "Minor")

        if st.button("Predict Era", use_container_width=True):
            if lyrics_input.strip() == "":
                st.error("Please provide lyrics!")
            else:
                acoustic_energy_ratio = acousticness / (energy + 0.001)
                dance_valence_product = danceability * valence
                loudness_norm = (loudness + 60) / 60
                if tempo <= 90: tempo_bin = 0.0
                elif tempo <= 110: tempo_bin = 1.0
                elif tempo <= 130: tempo_bin = 2.0
                elif tempo <= 160: tempo_bin = 3.0
                else: tempo_bin = 4.0
                speech_acoustic_diff = speechiness - acousticness

                vec = [
                    danceability, energy, loudness, speechiness,
                    acousticness, instrumentalness, liveness, valence,
                    tempo, duration_ms,
                    acoustic_energy_ratio, dance_valence_product, loudness_norm,
                    tempo_bin, speech_acoustic_diff
                ]
                for k in range(12):
                    vec.append(1.0 if key == k else 0.0)
                for m in [0, 1]:
                    vec.append(1.0 if mode == m else 0.0)

                pred_era, probs, unimodal_probs = predict_with_audio(vec, {'lyrics': lyrics_input}) # Adjusted call
                
                st.success(f"### Predicted Era: **{pred_era}** ✨")
                
                # ── Model Interpretability ───────────────────────────────────────────
                st.markdown("### 🔍 Model Interpretability: Sub-Classifier Analysis")
                with st.expander("View Sub-Model Contributions", expanded=True):
                    col_a, col_t = st.columns(2)
                    
                    audio_top = max(unimodal_probs['audio'], key=unimodal_probs['audio'].get)
                    text_top = max(unimodal_probs['text'], key=unimodal_probs['text'].get)
                    
                    with col_a:
                        st.markdown(f"**🔈 Acoustic Sub-Model:**")
                        if unimodal_probs['audio_active']:
                            st.info(f"The acoustic path identifies stylistic convergence with the **{audio_top}** era based on the specific distribution of density and energy features.")
                        else:
                            st.info("Acoustic path is inactive for this specific model configuration.")
                    
                    with col_t:
                        st.markdown(f"**📝 Semantic Sub-Model:**")
                        if unimodal_probs['text_active']:
                            st.info(f"The semantic path independently classifies the lyrical input as **{text_top}** based on TF-IDF term weighting and sentiment variance.")
                        else:
                            st.info("Semantic path is inactive for this specific model configuration.")
                    
                    st.divider()
                    st.markdown(f"**⚖️ Final Decision Layer:**")
                    st.warning(f"The meta-classifier aggregated both sub-model outputs to confirm **{pred_era}**. (Confidence: {max(probs.values()):.1%})")

                # ── Radar Chart ──────────────────────────────────────────────
                st.markdown("#### 🌀 Stylistic Signature (Radar)")
                categories = ['Danceability', 'Energy', 'Acousticness', 'Valence', 'Speechiness']
                values = [danceability, energy, acousticness, valence, speechiness]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Current Song',
                    marker=dict(color='#c9a96e')
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Class Probabilities")
                prob_df = pd.DataFrame({
                    "Era": list(probs.keys()),
                    "Probability": list(probs.values())
                }).sort_values(by="Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Era"))

    with tab3:
        st.subheader("Evaluation Analysis")
        st.write("Comparing Unimodal (Text Only, Audio Only) vs Multimodal (Early & Late Fusion) Architectures.")

        st.write("### Model Performance Comparison (5-Fold CV)")
        st.dataframe(res_df.style.highlight_max(axis=0))

        if 'era_report' in metrics:
            st.write("### Per-Era Classification Report (Best Model)")
            era_df = pd.DataFrame(metrics['era_report']).T
            
            # format styling
            st.dataframe(era_df.style.format("{:.3f}").background_gradient(cmap='Blues'))

        col_img1, col_img2 = st.columns([1, 1])

        with col_img1:
            if os.path.exists('artifacts/confusion_matrix.png'):
                st.write("### Late Fusion Confusion Matrix")
                st.write("Analyzes which eras the model struggles to differentiate due to semantic or acoustic overlap.")
                img = Image.open('artifacts/confusion_matrix.png')
                st.image(img, use_column_width=True)

        with col_img2:
            if os.path.exists('artifacts/era_differences.png'):
                st.write("### Audio Feature Importance & Shift")
                st.write("Visualizes the distributions of numerical acoustic features across stylistic periods.")
                img2 = Image.open('artifacts/era_differences.png')
                st.image(img2, use_column_width=True)

        st.write("### Theoretical Deep Dive")
        st.markdown("""
        **Sparse vs Dense Matrix Representations:**  
        The textual side relies on TF-IDF word frequency parsing, generating high dimensional, extremely sparse arrays where most data points are zero.
        Numerical audio parameters passed through standard scaling are dense, lower dimensional vectors. 
        
        **Fusion Methodologies:**  
        * **Early Fusion:** Concatenates Audio vectors with sparse text vectors. Highly prone to the Curse of Dimensionality mismatch where text sparsity drowns out numerical density.
        * **Late Fusion:** Text paths hit a Random Forest (resistant to high dimensional tree plotting), while Audio hits SVM mapping. Meta-layers aggregate class probabilities successfully avoiding structural data imbalance.
        """)
