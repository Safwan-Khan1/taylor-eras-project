import os
import re
import time
import pandas as pd
import lyricsgenius

# ==============================================================================
# INSTRUCTIONS: 
# The script requires a Genius API Access Token. We have automatically inserted 
# the token you provided below. If it ever expires, replace it here.
# ==============================================================================
GENIUS_ACCESS_TOKEN = "V77u6m2TuuEE6bdh3CQd0Qzi89FVAQn1OXe4qgMlc6OoQAJ7XB8Vtvai1T-qKu6W"

# 12 specific songs per era
ERAS_MAPPING = {
    "Country Era": {
        "album": "Taylor Swift (2006) / Fearless",
        "songs": [
            "Tim McGraw", "Picture To Burn", "Teardrops On My Guitar",
            "A Place In This World", "Cold As You", "Our Song",
            "Fearless", "Fifteen", "Love Story", 
            "White Horse", "You Belong With Me", "Breathe"
        ]
    },
    "Pop Era": {
        "album": "1989",
        "songs": [
            "Welcome To New York", "Blank Space", "Style", 
            "Out Of The Woods", "All You Had To Do Was Stay", "Shake It Off",
            "I Wish You Would", "Bad Blood", "Wildest Dreams", 
            "How You Get The Girl", "This Love", "Clean"
        ]
    },
    "Dark Pop Era": {
        "album": "Reputation",
        "songs": [
            "...Ready For It?", "End Game", "I Did Something Bad",
            "Don't Blame Me", "Delicate", "Look What You Made Me Do",
            "So It Goes...", "Gorgeous", "Getaway Car", 
            "King Of My Heart", "Dancing With Our Hands Tied", "Dress"
        ]
    },
    "Indie/Folk Era": {
        "album": "Folklore",
        "songs": [
            "the 1", "cardigan", "the last great american dynasty", 
            "exile", "my tears ricochet", "mirrorball", 
            "seven", "august", "this is me trying", 
            "illicit affairs", "invisible string", "mad woman"
        ]
    }
}

def clean_lyrics(lyrics):
    """
    Cleans the raw lyrics returned by lyricsgenius according to specifications.
    """
    if not lyrics:
        return ""
        
    # Remove the first line (lyricsgenius usually includes the title at the top, like "Tim McGraw Lyrics")
    lines = lyrics.split('\n')
    if len(lines) > 0 and 'Lyrics' in lines[0]:
        lines = lines[1:]
    lyrics = '\n'.join(lines)
    
    # Remove section identifiers like [Chorus], [Verse 1], etc.
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    
    # Clean ending embeddings like "120Embed" or "Embed"
    lyrics = re.sub(r'\d*Embed$', '', lyrics)
    lyrics = re.sub(r'Embed$', '', lyrics)
    
    # Remove extra whitespace and trailing spaces
    lyrics = re.sub(r'\n+', '\n', lyrics)
    lyrics = lyrics.strip()
    
    return lyrics

def generate_dataset():
    """
    Main orchestrating function to search for songs, scrape via lyricsgenius, clean, and build dataset.
    """
    dataset_records = []
    
    print("Starting Genius Lyrics Collection Pipeline using LyricsGenius...\n")
    
    # Using lyricsgenius to perfectly handle Genius HTML structure changes
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, retries=3, timeout=15)
    genius.remove_dict_tracks = True
    
    for era, details in ERAS_MAPPING.items():
        album = details["album"]
        songs = details["songs"]
        
        print(f"=== Scraping {era} ({album}) ===")
        
        for song in songs:
            print(f"Attempting to fetch: '{song}'...", end=" ")
            
            try:
                # Search and get lyrics natively using the library
                fetched_song = genius.search_song(song, "Taylor Swift", get_full_info=False)
                
                if fetched_song and fetched_song.lyrics:
                    cleaned = clean_lyrics(fetched_song.lyrics)
                    
                    dataset_records.append({
                        "song_name": song,
                        "album": album,
                        "era": era,
                        "lyrics": cleaned
                    })
                    print("SUCCESS")
                else:
                    print("FAILED (Could not extract lyrics)")
            except Exception as e:
                print(f"FAILED (Error: {e})")
            
            # Brief pause to respect rate limits
            time.sleep(1)
            
        print() # empty line for formatting
        
    return pd.DataFrame(dataset_records)

if __name__ == "__main__":
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    df_lyrics = generate_dataset()
    
    output_filepath = os.path.join(output_dir, "taylor_lyrics_clean.csv")
    df_lyrics.to_csv(output_filepath, index=False)
    
    print(f"\n=============================================")
    print(f"Dataset successfully created and saved!")
    print(f"Total songs collected: {len(df_lyrics)}")
    print(f"Location: {output_filepath}")
    print(f"=============================================")
