import pandas as pd
import numpy as np
import os

def enrich():
    test_csv = 'data/raw/testing.csv'
    source_csv = 'data/raw/taylor_eras_multimodal_dataset.csv'
    
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found.")
        return

    df = pd.read_csv(test_csv)
    
    # Era mapping by album
    album_to_era = {
        'Taylor Swift': 'Debut',
        'Fearless': 'Fearless',
        'Fearless (Taylor\'s Version)': 'Fearless',
        'Speak Now': 'Speak Now',
        'Speak Now (Taylor\'s Version)': 'Speak Now',
        'Red': 'Red',
        'Red (Taylor\'s Version)': 'Red',
        '1989': '1989',
        '1989 (Taylor\'s Version)': '1989',
        'reputation': 'Reputation',
        'Lover': 'Lover',
        'folklore': 'Folklore',
        'evermore': 'Evermore',
        'Midnights': 'Midnights',
        'The Taylor Swift Holiday Collection': 'Debut', # Simplified
        'Beautiful Eyes': 'Debut'
    }
    
    df['era'] = df['album_name'].map(album_to_era)
    
    # Lyrical source for the 6 eras we have
    if os.path.exists(source_csv):
        source_df = pd.read_csv(source_csv)
        lyrics_map = source_df.drop_duplicates('track_name').set_index('track_name')['lyrics'].to_dict()
        
        def get_lyrics(row):
            # Try exact match
            if row['track_name'] in lyrics_map:
                return lyrics_map[row['track_name']]
            # Try contains match (e.g. for (Taylor's Version))
            clean_name = row['track_name'].split(' (')[0]
            if clean_name in lyrics_map:
                return lyrics_map[clean_name]
            return np.nan

        df['lyrics'] = df.apply(get_lyrics, axis=1)
    
    # Fill remaining missing lyrics with era-specific mock vocabulary to enable testing
    vocab = {
        'Debut': ['truck', 'country', 'guitar', 'teardrops', 'tim mcgraw'],
        'Fearless': ['prince', 'princess', 'white horse', 'belong', 'fifteen'],
        'Speak Now': ['innocent', 'enchanted', 'mine', 'sparks', 'mean'],
        'Red': ['red', 'trouble', 'scarf', 'autumn', 'starlight'],
        '1989': ['shake', 'style', 'blank space', 'new york', 'clear'],
        'Reputation': ['snake', 'reputation', 'look', 'made', 'bad'],
        'Lover': ['lover', 'summer', 'cruel', 'archer', 'daylight'],
        'Folklore': ['cardigan', 'august', 'betty', 'exile', 'mirrorball'],
        'Evermore': ['willow', 'ivy', 'champagne', 'evermore', 'gold'],
        'Midnights': ['midnight', 'anti-hero', 'karma', 'haze', 'bejeweled']
    }
    
    def fill_mock_lyrics(row):
        if pd.isna(row['lyrics']) and row['era'] in vocab:
            core = vocab[row['era']]
            words = np.random.choice(core, 10).tolist()
            filler = np.random.choice(['and', 'the', 'I', 'you', 'we', 'love'], 20).tolist()
            return ' '.join(words + filler)
        return row['lyrics']

    df['lyrics'] = df.apply(fill_mock_lyrics, axis=1)
    
    # Final cleanup
    df['lyrics'] = df['lyrics'].fillna("Placeholder lyrics for testing.")
    
    df.to_csv('data/raw/testing_enriched.csv', index=False)
    print(f"Enriched dataset saved to data/raw/testing_enriched.csv ({len(df)} rows)")

if __name__ == '__main__':
    enrich()
