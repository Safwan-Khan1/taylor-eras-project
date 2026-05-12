import pandas as pd
import json

df = pd.read_csv('data/raw/taylor_eras_multimodal_dataset.csv')

picks = [
    'Shake It Off', 'Anti-Hero', 'Love Story', 'Blank Space',
    'cardigan', 'Lover', 'Our Song', 'august', 'Delicate', 'All Too Well'
]

cols = [
    'track_name','era','danceability','energy','loudness','speechiness',
    'acousticness','instrumentalness','liveness','valence','tempo',
    'duration_ms','key','mode','lyrics'
]

subset = df[df['track_name'].isin(picks)][cols].drop_duplicates('track_name')

records = []
for idx, r in subset.iterrows():
    rec = {
        'track_name': str(r['track_name']),
        'era': str(r['era']),
        'danceability': float(r['danceability']),
        'energy': float(r['energy']),
        'loudness': float(r['loudness']),
        'speechiness': float(r['speechiness']),
        'acousticness': float(r['acousticness']),
        'instrumentalness': float(r['instrumentalness']),
        'liveness': float(r['liveness']),
        'valence': float(r['valence']),
        'tempo': float(r['tempo']),
        'duration_ms': int(r['duration_ms']),
        'key': int(r['key']),
        'mode': int(r['mode']),
        'lyrics': str(r['lyrics'])[:500]
    }
    records.append(rec)

with open('test_songs_data.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2)

print(f"Saved {len(records)} songs:")
for s in records:
    print(f"  {s['track_name']} | {s['era']} | dance={s['danceability']} energy={s['energy']} loud={s['loudness']}")
