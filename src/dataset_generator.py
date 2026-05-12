import pandas as pd
import numpy as np
import os

def create_mock_dataset(filename="data/raw/taylor_swift_eras_placeholder.csv"):
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    eras = {
        'Country Era': ['Taylor Swift', 'Fearless'],
        'Pop Era': ['1989', 'Lover'],
        'Dark Pop Era': ['Reputation'],
        'Indie/Folk Era': ['Folklore', 'Evermore'],
        'Re-record Era': ['Midnights', 'TTPD']
    }
    
    record_template = []
    
    vocab = {
        'Country Era': ['truck', 'guitar', 'teardrops', 'tim mcgraw', 'our song', 'high school', 'kissing'],
        'Pop Era': ['beat', 'shake', 'style', 'blank space', 'cruel summer', 'new york', 'heartbeat'],
        'Dark Pop Era': ['snake', 'revenge', 'drama', 'look what you made me do', 'dark', 'crimes', 'blood'],
        'Indie/Folk Era': ['cardigan', 'woods', 'scarars', 'ghosts', 'tears', 'lakes', 'exile'],
        'Re-record Era': ['wine', 'midnight', 'karma', 'haze', 'clock', 'vigilante', 'mastermind']
    }
    
    # 20 songs per era
    for era_name, albums in eras.items():
        for i in range(20):
            album = np.random.choice(albums)
            song_name = f"{album} Track {i+1}"
            
            # Simulated audio features
            if era_name == 'Country Era':
                danceability = np.random.normal(0.5, 0.1)
                energy = np.random.normal(0.6, 0.1)
                valence = np.random.normal(0.6, 0.1)
                tempo = np.random.normal(110, 15)
                loudness = np.random.normal(-6, 1.5)
                acousticness = np.random.normal(0.4, 0.1)
                speechiness = np.random.normal(0.05, 0.02)
            elif era_name == 'Pop Era':
                danceability = np.random.normal(0.75, 0.1)
                energy = np.random.normal(0.8, 0.1)
                valence = np.random.normal(0.8, 0.1)
                tempo = np.random.normal(120, 10)
                loudness = np.random.normal(-4, 1.0)
                acousticness = np.random.normal(0.1, 0.05)
                speechiness = np.random.normal(0.06, 0.02)
            elif era_name == 'Dark Pop Era':
                danceability = np.random.normal(0.65, 0.1)
                energy = np.random.normal(0.85, 0.1)
                valence = np.random.normal(0.3, 0.1)
                tempo = np.random.normal(130, 20)
                loudness = np.random.normal(-3, 1.0)
                acousticness = np.random.normal(0.05, 0.05)
                speechiness = np.random.normal(0.12, 0.05)
            elif era_name == 'Indie/Folk Era':
                danceability = np.random.normal(0.4, 0.1)
                energy = np.random.normal(0.3, 0.1)
                valence = np.random.normal(0.4, 0.1)
                tempo = np.random.normal(100, 20)
                loudness = np.random.normal(-10, 2.0)
                acousticness = np.random.normal(0.8, 0.15)
                speechiness = np.random.normal(0.04, 0.01)
            elif era_name == 'Re-record Era':
                danceability = np.random.normal(0.6, 0.1)
                energy = np.random.normal(0.5, 0.1)
                valence = np.random.normal(0.4, 0.1)
                tempo = np.random.normal(105, 10)
                loudness = np.random.normal(-8, 1.5)
                acousticness = np.random.normal(0.5, 0.1)
                speechiness = np.random.normal(0.08, 0.02)
                
            words = np.random.choice(vocab[era_name], 15).tolist()
            filler = np.random.choice(['and', 'the', 'I', 'you', 'we', 'love', 'time', 'know'], 30).tolist()
            np.random.shuffle(words)
            np.random.shuffle(filler)
            lyrics = ' '.join(words + filler)
            
            record_template.append({
                'song_name': song_name,
                'album': album,
                'era': era_name,
                'lyrics': lyrics,
                'danceability': max(0, min(1, danceability)),
                'energy': max(0, min(1, energy)),
                'valence': max(0, min(1, valence)),
                'tempo': max(60, min(200, tempo)),
                'loudness': max(-20, min(0, loudness)),
                'acousticness': max(0, min(1, acousticness)),
                'speechiness': max(0, min(1, speechiness))
            })
            
    df = pd.DataFrame(record_template)
    df.to_csv(filename, index=False)
    print(f"Mock placeholder dataset generated at {filename} with {len(df)} songs.")
    return df

if __name__ == '__main__':
    create_mock_dataset()
