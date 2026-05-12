import lyricsgenius
import re
import time

class GeniusCollector:
    """
    A class to handle lyrics extraction using the Genius API.
    Cleans raw lyrics text to remove HTML artifacts and headers.
    """
    def __init__(self, access_token):
        if not access_token:
            raise ValueError("Genius Access Token must be provided.")
            
        print("Initializing Genius API...")
        # Automatically retries on timeouts and removes section headers
        self.genius = lyricsgenius.Genius(access_token, retries=3, timeout=15)
        self.genius.remove_dict_tracks = True

    def clean_lyrics(self, lyrics):
        """
        Cleans the raw lyrics returned by lyricsgenius according to specifications.
        """
        if not lyrics:
            return ""
            
        # 1. Remove the first line (lyricsgenius usually includes the title at the top, e.g. "Tim McGraw Lyrics")
        lines = lyrics.split('\n')
        if len(lines) > 0 and 'Lyrics' in lines[0]:
            lines = lines[1:]
        lyrics = '\n'.join(lines)
        
        # 2. Remove section identifiers like [Chorus], [Verse 1], etc.
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        
        # 3. Clean ending embedding tags sometimes added by the page structure like "120Embed" or "Embed"
        lyrics = re.sub(r'\d*Embed$', '', lyrics)
        lyrics = re.sub(r'Embed$', '', lyrics)
        
        # 4. Remove extra whitespace and trailing spaces
        lyrics = re.sub(r'\n+', '\n', lyrics)
        lyrics = lyrics.strip()
        
        return lyrics

    def fetch_lyrics(self, song_name, artist_name="Taylor Swift"):
        """
        Searches for a specific song and returns the cleaned lyrics string.
        """
        try:
            # get_full_info=False speeds up extraction drastically
            fetched_song = self.genius.search_song(song_name, artist_name, get_full_info=False)
            
            # brief pause to stop api blocking
            time.sleep(0.5)
            
            if fetched_song and fetched_song.lyrics:
                return self.clean_lyrics(fetched_song.lyrics)
            else:
                return None
                
        except Exception as e:
            print(f"  [ERROR] Genius API failed for '{song_name}': {e}")
            return None
