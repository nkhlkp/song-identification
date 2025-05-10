import yt_dlp
import streamlit as st
import numpy as np
from matplotlib import mlab
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure, binary_erosion
import matplotlib.pyplot as plt
import librosa
import os
from collections import defaultdict, Counter
import pickle
import sys
from pydub import AudioSegment
import re
import tempfile

# Create necessary directories
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------------------------
def sanitize_filename(filename):
    # Remove characters not allowed in filenames
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def download_best_audio_as_mp3(video_url, save_path=DOWNLOADS_DIR):
    ydl_opts = {
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

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

# Rest of your audio processing functions remain unchanged
# ---------------------------------------------------------------------------------------------

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

# Other functions remain the same
# ...

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
    
    # Other methods remain the same
    
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
    
    # Rest of the methods remain the same

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

# YouTube URL form
with st.form("get_link"):
    video_link = st.text_input("Enter the YouTube URL of the song:")
    submitted = st.form_submit_button("Upload Song")
    if submitted and video_link:
        download_best_audio_as_mp3(video_link, DOWNLOADS_DIR)
        raw_title = get_video_title(video_link, DOWNLOADS_DIR)
        video_title = sanitize_filename(raw_title)
        video_file_path = os.path.join(DOWNLOADS_DIR, f"{video_title}.mp3")
        
        if os.path.exists(video_file_path):
            fingerprinter.add_song(video_file_path, video_title)
            fingerprinter.save_database()
            st.success(f"Song '{video_title}' uploaded and fingerprinted.")
        else:
            st.error("MP3 file not found after download.")

# Record audio form
with st.form("get_sample_from_microphone"):
    sample_recorded_audio = st.audio_input(label="Record Audio")
    submitted = st.form_submit_button("Submit")
    if submitted and sample_recorded_audio:
        # Save recorded audio to a temporary file
        temp_wav = os.path.join(tempfile.gettempdir(), "recorded_audio.wav")
        temp_mp3 = os.path.join(tempfile.gettempdir(), "sample_recorded_audio.mp3")
        
        with open(temp_wav, "wb") as f:
            f.write(sample_recorded_audio.getbuffer())
        
        # Convert wav to mp3
        sound = AudioSegment.from_wav(temp_wav)
        sound.export(temp_mp3, format="mp3")
        
        # Identify the song
        results = fingerprinter.identify_sample(temp_mp3)
        
        if results:
            st.success("Match found!")
        else:
            st.warning("No matches found in the database.")

# Upload audio file form
with st.form("get_sample_from_file"):
    sample_uploaded_audio = st.file_uploader(label="Upload Audio")
    submitted = st.form_submit_button("Submit")
    if submitted and sample_uploaded_audio:
        # Save uploaded audio to a temporary file
        temp_file = os.path.join(tempfile.gettempdir(), "uploaded_audio.mp3")
        with open(temp_file, "wb") as f:
            f.write(sample_uploaded_audio.getbuffer())
        
        # Identify the song
        results = fingerprinter.identify_sample(temp_file)
        
        if results:
            st.success("Match found!")
        else:
            st.warning("No matches found in the database.")
