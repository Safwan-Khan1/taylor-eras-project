import pandas as pd, json

df = pd.read_csv('data/raw/taylor_eras_multimodal_dataset.csv')
print('Songs per era:')
print(df['era'].value_counts().to_string())
print()

audio_cols = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
print('Audio feature stats:')
print(df[audio_cols].describe().round(3).to_string())
print()

with open('artifacts/eval_metrics.json') as f:
    m = json.load(f)
print('Current results:')
for k, v in m['results'].items():
    acc = v['accuracy']
    prec = v['precision']
    rec = v['recall']
    f1 = v['f1_score']
    print(f'  {k}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}')
