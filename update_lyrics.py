import pandas as pd
import numpy as np
import os

# Define the vocabulary for each era (same as in dataset_generator.py)
vocab = {
    'Country Era': ['truck', 'guitar', 'teardrops', 'tim mcgraw', 'our song', 'high school', 'kissing'],
    'Pop Era': ['beat', 'shake', 'style', 'blank space', 'cruel summer', 'new york', 'heartbeat'],
    'Dark Pop Era': ['snake', 'revenge', 'drama', 'look what you made me do', 'dark', 'crimes', 'blood'],
    'Indie/Folk Era': ['cardigan', 'woods', 'scarars', 'ghosts', 'tears', 'lakes', 'exile'],
}

csv_path = 'data/raw/taylor_swift_eras_data.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    
    def generate_lyrics(era):
        if era in vocab:
            env_words = vocab[era]
            words = np.random.choice(env_words, 15).tolist()
            filler = np.random.choice(['and', 'the', 'I', 'you', 'we', 'love', 'time', 'know'], 30).tolist()
            np.random.shuffle(words)
            np.random.shuffle(filler)
            return ' '.join(words + filler)
        return "Placeholder lyrics..."

    df['lyrics'] = df['era'].apply(generate_lyrics)
    df.to_csv(csv_path, index=False)
    print(f"Updated lyrics for {len(df)} songs in {csv_path}")
else:
    print(f"Error: {csv_path} not found.")
