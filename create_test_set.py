import pandas as pd

def create_test_set():
    # Load the training set
    df_train = pd.read_csv('data/raw/taylor_eras_multimodal_dataset.csv')
    train_songs = set(df_train['track_name'].str.lower())
    
    # Load potential sources
    sources = [
        'data/raw/testing_enriched.csv',
        'data/raw/taylor_merged_df.csv',
        'data/raw/taylor_lyric_df_processed.csv'
    ]
    
    target_eras = ['1989', 'Reputation', 'Lover', 'Folklore', 'Evermore', 'Midnights']
    
    all_unseen = []
    
    for src in sources:
        df = pd.read_csv(src)
        # column name normalization
        if 'track_name' in df.columns:
            name_col = 'track_name'
        elif 'song_title' in df.columns:
            name_col = 'song_title'
        else:
            continue
            
        if 'era' not in df.columns and 'album_name' in df.columns:
            album_to_era = {
                '1989': '1989', 'reputation': 'Reputation', 'Lover': 'Lover',
                'folklore': 'Folklore', 'evermore': 'Evermore', 'Midnights': 'Midnights'
            }
            df['era'] = df['album_name'].map(album_to_era)
            
        if 'era' not in df.columns:
            continue
            
        # Filter for target eras and unseen songs
        unseen = df[df['era'].isin(target_eras) & ~df[name_col].str.lower().isin(train_songs)]
        if not unseen.empty:
            # Standardize for the test set
            subset = unseen.copy()
            subset = subset.rename(columns={name_col: 'track_name'})
            all_unseen.append(subset)
            
    if not all_unseen:
        print("No unseen songs found in the 6 targeted eras across source files.")
        # Alternative: Create a hold-out test set from the 117 songs
        print("Creating hold-out test set from existing multimodal dataset (20% sample).")
        test_df = df_train.sample(frac=0.2, random_state=42)
        train_df = df_train.drop(test_df.index)
        
        test_df.to_csv('data/processed/test_dataset.csv', index=False)
        train_df.to_csv('data/raw/taylor_eras_multimodal_dataset.csv', index=False)
        print(f"Created hold-out test set: {len(test_df)} songs. Train set reduced to {len(train_df)} songs.")
        return
        
    final_test = pd.concat(all_unseen, ignore_index=True)
    final_test['track_name_lower'] = final_test['track_name'].str.lower()
    final_test = final_test.drop_duplicates(subset=['track_name_lower'])
    final_test = final_test.drop(columns=['track_name_lower'])
    
    # Save
    final_test.to_csv('data/processed/test_dataset.csv', index=False)
    print(f"Created testing dataset with {len(final_test)} unique unseen songs.")
    print(f"Eras in test set:\n{final_test['era'].value_counts()}")

if __name__ == "__main__":
    create_test_set()
