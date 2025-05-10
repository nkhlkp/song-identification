import yt_dlp
import streamlit as st
import numpy as np
from matplotlib import mlab
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure, binary_erosion
import matplotlib.pyplot as plt
import librosa
import os
import sys
import subprocess
from collections import defaultdict, Counter
import pickle
import re
from pydub import AudioSegment
import tempfile

# Create necessary directories
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------------------------
def sanitize_filename(filename):
    # Remove characters not allowed in filenames
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def download_best_audio_as_mp3(video_url, save_path=DOWNLOADS_DIR):
    # Clear the cache first (helps with 403 errors)
    try:
        subprocess.run(["yt-dlp", "--rm-cache-dir"], check=False)
    except Exception as e:
        st.warning(f"Could not clear cache: {e}")
    
    # Set up options with multiple fixes for 403 errors
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
        # Add user agent (helps bypass some restrictions)
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        # Force IPv4 (recent fix for 403 errors)
        'force_ipv4': True,
        # Add verbose output for debugging
        'verbose': True,
        # Add referer (can help with some restrictions)
        'referer': 'https://www.youtube.com/',
        # Retry on HTTP errors
        'retries': 10,
        # Sleep between retries
        'sleep_interval': 5,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_title = info.get('title', 'unknown_title')
            sanitized_title = sanitize_filename(video_title)
            expected_path = os.path.join(save_path, f"{sanitized_title}.mp3")
            return expected_path
    except Exception as e:
        st.error(f"First attempt failed: {e}")
        
        # Try with IPv6 if IPv4 failed
        ydl_opts['force_ipv4'] = False
        ydl_opts['force_ipv6'] = True
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                video_title = info.get('title', 'unknown_title')
                sanitized_title = sanitize_filename(video_title)
                expected_path = os.path.join(save_path, f"{sanitized_title}.mp3")
                return expected_path
        except Exception as e2:
            st.error(f"Second attempt failed: {e2}")
            
            # Try with specific format as last resort
            ydl_opts['force_ipv6'] = False
            ydl_opts['format'] = '140'  # Common audio format on YouTube
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    video_title = info.get('title', 'unknown_title')
                    sanitized_title = sanitize_filename(video_title)
                    expected_path = os.path.join(save_path, f"{sanitized_title}.mp3")
                    return expected_path
            except Exception as e3:
                st.error(f"All download attempts failed. Last error: {e3}")
                return None

def get_video_title(video_url, save_path=DOWNLOADS_DIR):
    ydl_opts = {
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        video_title = info_dict.get('title', None)
        return video_title

# SpotDL Integration
def install_spotdl_if_needed():
    """Check if spotdl is installed and install if needed"""
    try:
        subprocess.run(["spotdl", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.info("Installing spotDL (this may take a moment)...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "spotdl"], check=True)
            # Also install ffmpeg if needed
            subprocess.run(["spotdl", "--download-ffmpeg"], check=True)
            return True
        except Exception as e:
            st.error(f"Failed to install spotDL: {e}")
            return False

def download_from_spotify(spotify_url, save_path=DOWNLOADS_DIR):
    """Download a song from Spotify and return the path to the downloaded file"""
    
    if not install_spotdl_if_needed():
        return None
        
    try:
        # Create a temporary file to capture the output
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp_path = tmp.name
            
        # Run spotdl with the Spotify URL
        process = subprocess.run(
            ["spotdl", "download", spotify_url, "--output", save_path, "--output-format", "mp3"],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        if process.returncode != 0:
            st.error(f"spotDL error: {process.stderr}")
            return None
            
        # Parse the output to find the downloaded file path
        output_lines = process.stdout.split('\n')
        downloaded_files = []
        
        for line in output_lines:
            if "Downloaded" in line and ".mp3" in line:
                # Extract the file path - this pattern may need adjustment based on spotdl's output format
                file_path = line.split("Downloaded")[1].strip()
                if os.path.exists(file_path):
                    downloaded_files.append(file_path)
                    
        # If we found files, return the first one (or you could return all)
        if downloaded_files:
            return downloaded_files[0]
            
        # If we couldn't find the file in the output, search the directory
        # This is a fallback in case the output parsing fails
        for file in os.listdir(save_path):
            if file.endswith('.mp3') and os.path.getmtime(os.path.join(save_path, file)) > os.path.getmtime(tmp_path):
                return os.path.join(save_path, file)
                
        return None
        
    except Exception as e:
        st.error(f"Error downloading from Spotify: {e}")
        return None
        
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def get_song_info_from_spotify(spotify_url):
    """Get song information from a Spotify URL without downloading"""
    
    if not install_spotdl_if_needed():
        return None, None
        
    try:
        # Run spotdl with the Spotify URL in search mode
        process = subprocess.run(
            ["spotdl", "query", spotify_url],
            capture_output=True,
            text=True,
            check=False
        )
        
        if process.returncode != 0:
            st.error(f"spotDL error: {process.stderr}")
            return None, None
            
        # Parse the output to find song info
        output = process.stdout
        
        # Extract title and artist - this pattern needs adjustment based on spotdl's output
        title = None
        artist = None
        
        for line in output.split('\n'):
            if "Title:" in line:
                title = line.split("Title:")[1].strip()
            if "Artist:" in line:
                artist = line.split("Artist:")[1].strip()
                
        return title, artist
        
    except Exception as e:
        st.error(f"Error getting song info: {e}")
        return None, None

# Audio processing functions
def load_and_process_audio(file_path, sr=22050, duration=None):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=duration)
    
    # Create spectrogram
    n_fft = 2048
    hop_length = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    spectrogram = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(S, sr=sr, hop_length=hop_length)
    
    return spectrogram, freqs, times

def create_constellation_map(spectrogram, freqs, times, threshold=75):
    # Find peaks in the spectrogram
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, 20)  # Larger neighborhood for better peak isolation
    
    # Apply maximum filter to find local maxima
    local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram
    
    # Apply threshold to remove background noise
    background = (spectrogram < threshold)
    
    # Combine local maxima and threshold to find peaks
    peaks = local_max & ~background
    
    # Extract peak coordinates
    peak_coords = np.where(peaks)
    peak_values = spectrogram[peak_coords]
    
    # Create constellation map as (time, frequency, amplitude) tuples
    constellation = []
    for i in range(len(peak_coords[0])):
        freq_idx = peak_coords[0][i]
        time_idx = peak_coords[1][i]
        constellation.append((times[time_idx], freqs[freq_idx], peak_values[i]))
    
    return constellation

def generate_hashes(constellation, fan_out=15, time_window=0.5):
    # Sort constellation points by time
    constellation.sort(key=lambda x: x[0])
    
    # Generate hashes
    hashes = []
    for i in range(len(constellation)):
        # Anchor point
        anchor_time, anchor_freq, _ = constellation[i]
        
        # Look at the next fan_out points within the time window
        for j in range(1, fan_out + 1):
            if i + j < len(constellation):
                target_time, target_freq, _ = constellation[i + j]
                
                # Check if the target is within the time window
                if target_time - anchor_time <= time_window:
                    # Create a hash: (anchor_freq, target_freq, delta_time)
                    freq_delta = target_freq - anchor_freq
                    time_delta = target_time - anchor_time
                    
                    # Create a hash string
                    hash_str = f"{int(anchor_freq)}|{int(target_freq)}|{time_delta:.6f}"
                    
                    # Store the hash with the anchor time
                    hashes.append((hash_str, anchor_time))
                else:
                    # If we've gone beyond the time window, break
                    break
    
    return hashes

class AudioFingerprinter:
    def __init__(self, database_path=None, mapping_path=None):
        self.database = {}
        self.song_mapping = {}
        
        # Load existing database if provided
        if database_path and mapping_path and os.path.exists(database_path) and os.path.exists(mapping_path):
            try:
                with open(database_path, "rb") as f:
                    self.database = pickle.load(f)
                with open(mapping_path, "rb") as f:
                    self.song_mapping = pickle.load(f)
                print(f"Loaded database with {len(self.database)} unique hashes and {len(self.song_mapping)} songs")
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def add_song(self, file_path, song_name):
        # Generate a unique ID for the song
        song_id = len(self.song_mapping) + 1
        
        # Add to song mapping
        self.song_mapping[song_id] = song_name
        
        # Process the audio
        spectrogram, freqs, times = load_and_process_audio(file_path)
        
        # Create constellation map
        constellation = create_constellation_map(spectrogram, freqs, times)
        
        # Generate hashes
        hashes = generate_hashes(constellation)
        
        # Add hashes to database
        for hash_str, offset in hashes:
            if hash_str not in self.database:
                self.database[hash_str] = []
            self.database[hash_str].append((song_id, offset))
        
        return song_id
    
    def add_songs_from_directory(self, directory_path):
        added_songs = []
        
        # Get all audio files in the directory
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        audio_files = [f for f in os.listdir(directory_path)
                      if os.path.isfile(os.path.join(directory_path, f))
                      and os.path.splitext(f)[1].lower() in audio_extensions]
        
        # Add each song
        for audio_file in audio_files:
            file_path = os.path.join(directory_path, audio_file)
            song_name = os.path.splitext(os.path.basename(audio_file))[0]
            try:
                song_id = self.add_song(file_path, song_name)
                added_songs.append((song_id, song_name))
            except Exception as e:
                print(f"Error adding {audio_file}: {e}")
                
        return added_songs
    
    def identify_sample(self, file_path, duration=10):
        # Process the sample audio
        spectrogram, freqs, times = load_and_process_audio(file_path, duration=duration)
        
        # Create constellation map
        constellation = create_constellation_map(spectrogram, freqs, times)
        
        # Generate hashes
        sample_hashes = generate_hashes(constellation)
        
        # Match against database
        matches = defaultdict(list)
        
        for hash_str, sample_offset in sample_hashes:
            if hash_str in self.database:
                for song_id, song_offset in self.database[hash_str]:
                    # Calculate the time difference
                    time_diff = song_offset - sample_offset
                    matches[song_id].append(time_diff)
        
        # Count the number of matching hashes for each song
        match_counts = {song_id: len(time_diffs) for song_id, time_diffs in matches.items()}
        
        # Find the song with the most matches
        if not match_counts:
            return []
        
        # Sort songs by number of matches (descending)
        sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top matches with confidence scores
        results = []
        total_matches = sum(count for _, count in sorted_matches)
        
        for song_id, count in sorted_matches[:5]:  # Return top 5 matches
            confidence = count / total_matches if total_matches > 0 else 0
            song_name = self.song_mapping.get(song_id, f"Unknown Song ({song_id})")
            results.append({
                "song_id": song_id,
                "song_name": song_name,
                "matches": count,
                "confidence": confidence
            })
        
        return results
    
    def save_database(self, database_path="fingerprint_database.pkl", mapping_path="song_mapping.pkl"):
        # Save the database to disk
        with open(database_path, "wb") as f:
            pickle.dump(self.database, f)
        
        # Save the song mapping to disk
        with open(mapping_path, "wb") as f:
            pickle.dump(self.song_mapping, f)
        
        print(f"Saved database with {len(self.database)} unique hashes and {len(self.song_mapping)} songs")

# Function to clear all files in the downloads directory
def clear_downloads_directory():
    try:
        for filename in os.listdir(DOWNLOADS_DIR):
            file_path = os.path.join(DOWNLOADS_DIR, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
        st.success("All files in downloads directory have been removed.")
    except Exception as e:
        st.error(f"Error clearing downloads directory: {e}")

# ---------------------------------------------------------------------------------------------

st.title("Song Identification App")

# Initialize or load database
if os.path.exists("fingerprint_database.pkl") and os.path.exists("song_mapping.pkl"):
    fingerprinter = AudioFingerprinter("fingerprint_database.pkl", "song_mapping.pkl")
else:
    fingerprinter = AudioFingerprinter()

# Add songs from downloads directory if it exists
if st.button("Load existing songs"):
    if os.path.exists(DOWNLOADS_DIR):
        fingerprinter.add_songs_from_directory(DOWNLOADS_DIR)
        fingerprinter.save_database()
        st.success("Songs loaded successfully!")
    else:
        st.warning(f"Directory {DOWNLOADS_DIR} does not exist yet.")

# Clear downloads directory button
if st.button("Clear Downloads Directory"):
    clear_downloads_directory()

# Create tabs for different download methods
tab1, tab2, tab3, tab4 = st.tabs(["YouTube Download", "Spotify Download", "Record Audio", "Upload Audio"])

# YouTube URL form
with tab1:
    st.subheader("Download from YouTube")
    with st.form("get_link"):
        video_link = st.text_input("Enter the YouTube URL of the song:")
        submitted = st.form_submit_button("Upload Song")
        if submitted and video_link:
            with st.spinner("Downloading and processing audio..."):
                file_path = download_best_audio_as_mp3(video_link, DOWNLOADS_DIR)
                
                if file_path and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    video_title = os.path.basename(file_path).replace('.mp3', '')
                    fingerprinter.add_song(file_path, video_title)
                    fingerprinter.save_database()
                    st.success(f"Song '{video_title}' uploaded and fingerprinted.")
                else:
                    st.error("Failed to download the song due to YouTube restrictions.")
                    st.info("Try a different YouTube URL or check if the video is available in your region.")

# Spotify download form
with tab2:
    st.subheader("Download from Spotify")
    with st.form("spotify_download_form"):
        spotify_url = st.text_input("Enter Spotify URL (song, album, or playlist):")
        submitted = st.form_submit_button("Download")
        
        if submitted and spotify_url:
            if "spotify.com" not in spotify_url:
                st.error("Please enter a valid Spotify URL")
            else:
                with st.spinner("Downloading from Spotify..."):
                    # Get song info first
                    title, artist = get_song_info_from_spotify(spotify_url)
                    
                    if title and artist:
                        st.info(f"Found: {title} by {artist}")
                        
                    # Download the song
                    file_path = download_from_spotify(spotify_url, DOWNLOADS_DIR)
                    
                    if file_path and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        song_name = os.path.basename(file_path).replace('.mp3', '')
                        
                        # Add to your fingerprinter database
                        fingerprinter.add_song(file_path, song_name)
                        fingerprinter.save_database()
                        
                        st.success(f"Song '{song_name}' downloaded and fingerprinted.")
                    else:
                        st.error("Failed to download the song.")
                        st.info("Check if the Spotify URL is valid and accessible.")

# Record audio form
with tab3:
    st.subheader("Record Audio Sample")
    sample_recorded_audio = st.audio_input(label="Record Audio")
    if sample_recorded_audio:
        # Save recorded audio to a temporary file
        temp_wav = os.path.join(tempfile.gettempdir(), "recorded_audio.wav")
        temp_mp3 = os.path.join(tempfile.gettempdir(), "sample_recorded_audio.mp3")
        
        with open(temp_wav, "wb") as f:
            f.write(sample_recorded_audio.getbuffer())
        
        # Convert wav to mp3
        sound = AudioSegment.from_wav(temp_wav)
        sound.export(temp_mp3, format="mp3")
        
        # Identify the song
        with st.spinner("Identifying song..."):
            results = fingerprinter.identify_sample(temp_mp3)
            
            if results:
                st.success("Match found!")
                for i, result in enumerate(results, 1):
                    st.write(f"{i}. {result['song_name']} (Confidence: {result['confidence']:.2%})")
            else:
                st.warning("No matches found in the database.")

# Upload audio file form
with tab4:
    st.subheader("Upload Audio Sample")
    sample_uploaded_audio = st.file_uploader(label="Upload Audio")
    if sample_uploaded_audio:
        # Save uploaded audio to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), "uploaded_audio.mp3")
        with open(temp_file, "wb") as f:
            f.write(sample_uploaded_audio.getbuffer())
        
        # Identify the song
        with st.spinner("Identifying song..."):
            results = fingerprinter.identify_sample(temp_file)
            
            if results:
                st.success("Match found!")
                for i, result in enumerate(results, 1):
                    st.write(f"{i}. {result['song_name']} (Confidence: {result['confidence']:.2%})")
            else:
                st.warning("No matches found in the database.")

# Display songs in database
st.header("Songs in Database")

if st.button("Show all songs in database"):
    # Check if database exists and has songs
    if hasattr(fingerprinter, 'song_mapping') and fingerprinter.song_mapping:
        # Create a list of songs from your song_mapping dictionary
        songs_list = []
        for song_id, song_name in fingerprinter.song_mapping.items():
            songs_list.append({"ID": song_id, "Song Name": song_name})
        
        # Display as a dataframe
        import pandas as pd
        songs_df = pd.DataFrame(songs_list)
        st.dataframe(songs_df, use_container_width=True)
        
        # Also show the total count
        st.write(f"Total songs in database: {len(fingerprinter.song_mapping)}")
    else:
        st.info("No songs found in the database. Try adding some songs first!")
