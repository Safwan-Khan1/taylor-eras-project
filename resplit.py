import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

def create_new_split(seed=100):
    multi_path = 'data/raw/taylor_eras_multimodal_dataset.csv'
    df = pd.read_csv(multi_path)

    # New random split with a different seed
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=seed, 
        stratify=df['era']
    )

    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train_dataset.csv', index=False)
    test_df.to_csv('data/processed/test_dataset.csv', index=False)

    print(f"Created NEW Train/Test split using seed {seed}.")
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Update web app test lab
    test_list = test_df.to_dict(orient='records')
    with open('test_songs_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_list, f, indent=2)

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    create_new_split(seed)
