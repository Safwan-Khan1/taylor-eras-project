import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time

class SpotifyCollector:
    """
    A class to handle all interactions with the Spotify Web API.
    Extracts albums, tracklists, and detailed audio features.
    """
    def __init__(self, client_id, client_secret):
        if not client_id or not client_secret:
            raise ValueError("Spotify Client ID and Client Secret must be provided.")
            
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        # Adding retries to prevent random rate-limiting failures
        self.sp = spotipy.Spotify(auth_manager=auth_manager, retries=5, requests_timeout=10)
        
    def get_album_tracks_with_features(self, album_query):
        """
        Searches for an album by Taylor Swift, fetches all its tracks, 
        and retrieves audio features for each track.
        """
        print(f"Searching Spotify for Album: '{album_query}'...")
        
        # Search for the album specifically by Taylor Swift
        try:
            results = self.sp.search(q=f"album:{album_query} artist:Taylor Swift", type="album", limit=1)
        except Exception as e:
            print(f"  [ERROR] Spotify API failed while searching for album '{album_query}'.")
            print(f"  [Details]: {e}")
            return []
            
        albums = results.get("albums", {}).get("items", [])
        
        if not albums:
            print(f"  [ERROR] Album '{album_query}' not found on Spotify.")
            return []
            
        album_id = albums[0]["id"]
        album_name = albums[0]["name"]
        print(f"  [SUCCESS] Found Spotify Album Match: {album_name} (ID: {album_id})")
        
        # Fetch tracks from the album
        tracks = []
        try:
            track_results = self.sp.album_tracks(album_id)
            tracks.extend(track_results["items"])
            
            # Pagination for albums with >50 tracks
            while track_results["next"]:
                track_results = self.sp.next(track_results)
                tracks.extend(track_results["items"])
        except Exception as e:
            print(f"  [ERROR] Failed to fetch tracks for {album_name}: {e}")
            return []
            
        print(f"  --> Found {len(tracks)} tracks. Fetching audio features...")
        
        # Note: audio_features limit is 100 per request
        track_ids = [t["id"] for t in tracks]
        features = [None] * len(track_ids) # Initialize with None
        
        try:
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                batch_features = self.sp.audio_features(batch)
                
                if batch_features:
                    # Fill specifically the range of the batch
                    features[i:i+len(batch_features)] = batch_features
                
                # Sleep briefly to avoid aggressive rate limiting
                time.sleep(0.5) 
        except Exception as e:
            print(f"  [WARN] Spotify API restricted 'audio_features' endpoint (403).")
            print(f"  [Details]: {e}")
            # We keep 'features' as a list of None values and proceed
                
        # Combine track info and audio features
        tracks_data = []
        for t, f in zip(tracks, features):
            # If Spotify returned no features or we hit a 403, we provide placeholders
            f_val = f if f else {}
            
            tracks_data.append({
                "song_name": t["name"],
                "spotify_album": album_name,
                "danceability": f_val.get("danceability"),
                "energy": f_val.get("energy"),
                "loudness": f_val.get("loudness"),
                "tempo": f_val.get("tempo"),
                "valence": f_val.get("valence"),
                "acousticness": f_val.get("acousticness"),
                "speechiness": f_val.get("speechiness"),
                "instrumentalness": f_val.get("instrumentalness"),
                "liveness": f_val.get("liveness"),
                "duration_ms": f_val.get("duration_ms"),
                "key": f_val.get("key"),
                "mode": f_val.get("mode")
            })
            
        print(f"  [SUCCESS] Processed {len(tracks_data)} tracks for '{album_query}' (Audio features: {'Available' if any(features) else 'Restricted'}).")
        return tracks_data
