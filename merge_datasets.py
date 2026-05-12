import pandas as pd
import os

# Paths
MULTIMODAL_PATH = 'data/raw/taylor_eras_multimodal_dataset.csv'
MERGED_PATH = 'data/raw/taylor_merged_df.csv'
ENRICHED_PATH = 'data/raw/testing_enriched.csv' # I'll use this too as it had more Midnights/Evermore

def load_and_standardize():
    # 1. Load original multimodal
    df_multi = pd.read_csv(MULTIMODAL_PATH)
    print(f"Original multimodal: {df_multi.shape}")
    
    # Target eras
    target_eras = ['1989', 'Reputation', 'Lover', 'Folklore', 'Evermore', 'Midnights']
    
    # 2. Load taylor_merged_df.csv
    df_merged = pd.read_csv(MERGED_PATH)
    # Map albums to eras
    album_to_era = {
        '1989': '1989',
        'reputation': 'Reputation',
        'Lover': 'Lover',
        'folklore': 'Folklore'
        # evermore and midnights missing in this file
    }
    df_merged['era'] = df_merged['album_name'].map(album_to_era)
    df_merged = df_merged[df_merged['era'].isin(target_eras)]
    print(f"Candidate songs from merged_df (filtered for eras): {df_merged.shape}")
    
    # 3. Load testing_enriched.csv (has midnights/evermore)
    df_enriched = pd.read_csv(ENRICHED_PATH)
    df_enriched = df_enriched[df_enriched['era'].isin(target_eras)]
    print(f"Candidate songs from enriched_df (filtered for eras): {df_enriched.shape}")

    # Standardize columns
    # We want: track_name, album, era, lyrics, danceability, energy, loudness, 
    # speechiness, acousticness, instrumentalness, liveness, valence, tempo, 
    # duration_ms, key, mode
    
    def standardize_cols(df, type):
        if type == 'merged':
            rename_map = {'album_name': 'album'}
        elif type == 'enriched':
            rename_map = {'album_name': 'album'}
        else:
            rename_map = {}
            
        cols_needed = [
            'track_name', 'album', 'era', 'lyrics', 'danceability', 'energy', 
            'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms', 'key', 'mode'
        ]
        
        df = df.rename(columns=rename_map)
        # Ensure all columns exist
        for col in cols_needed:
            if col not in df.columns:
                df[col] = None
                
        return df[cols_needed]

    df_multi_std = standardize_cols(df_multi, 'multi')
    df_merged_std = standardize_cols(df_merged, 'merged')
    df_enriched_std = standardize_cols(df_enriched, 'enriched')
    
    # Combine
    final_df = pd.concat([df_multi_std, df_merged_std, df_enriched_std], ignore_index=True)
    
    # Deduplicate by track_name
    final_df['track_name_lower'] = final_df['track_name'].str.lower()
    final_df = final_df.drop_duplicates(subset=['track_name_lower'])
    final_df = final_df.drop(columns=['track_name_lower'])
    
    # Remove any rows where critical audio features or lyrics are missing
    final_df = final_df.dropna(subset=['lyrics', 'danceability', 'energy', 'era'])
    
    print(f"Final combined dataset: {final_df.shape}")
    print(f"Era distribution:\n{final_df['era'].value_counts()}")
    
    return final_df

if __name__ == "__main__":
    new_df = load_and_standardize()
    # Backup original
    if not os.path.exists(MULTIMODAL_PATH + '.bak'):
        os.rename(MULTIMODAL_PATH, MULTIMODAL_PATH + '.bak')
    
    new_df.to_csv(MULTIMODAL_PATH, index=False)
    print(f"Saved expanded dataset to {MULTIMODAL_PATH}")
