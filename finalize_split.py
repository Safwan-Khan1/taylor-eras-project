import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

def create_split_datasets():
    # Load the full multimodal dataset (117 songs)
    multi_path = 'data/raw/taylor_eras_multimodal_dataset.csv'
    if not os.path.exists(multi_path):
        print(f"Error: {multi_path} not found.")
        return

    df = pd.read_csv(multi_path)
    print(f"Original dataset size: {len(df)} songs.")

    # Stratified split to keep era distribution consistent
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['era']
    )

    # Save to processed directory
    os.makedirs('data/processed', exist_ok=True)
    train_df.to_csv('data/processed/train_dataset.csv', index=False)
    test_df.to_csv('data/processed/test_dataset.csv', index=False)

    print(f"Created Train Dataset: {len(train_df)} songs -> data/processed/train_dataset.csv")
    print(f"Created Test Dataset: {len(test_df)} songs -> data/processed/test_dataset.csv")

    # Era counts in test set
    print(f"\nTest Set Era Distribution:\n{test_df['era'].value_counts()}")

    # Generate test_songs_data.json for the App Testing Lab
    test_list = test_df.to_dict(orient='records')
    with open('test_songs_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_list, f, indent=2)
    print("Updated test_songs_data.json for the web application.")

if __name__ == "__main__":
    create_split_datasets()
