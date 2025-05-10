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
import urllib
from urllib.parse import urlparse, parse_qs

# Create necessary directories
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------------------------

def clean_youtube_url(url):
    parsed_url = urlparse(url)
    if 'youtu.be' in parsed_url.netloc:
        # Shortened URL format: https://youtu.be/VIDEO_ID
        video_id = parsed_url.path[1:]
    elif 'youtube.com' in parsed_url.netloc:
        # Standard URL format: https://www.youtube.com/watch?v=VIDEO_ID
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v', [None])[0]
    else:
        return None  # Not a YouTube URL

    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    else:
        return None
    

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


def find_peaks(spectrogram, freqs, times, amp_min=-40, neighbourhood_size=2):

    # Threshold the spectrogram
    threshold_spectrogram = np.copy(spectrogram)
    threshold_spectrogram[threshold_spectrogram < amp_min] = amp_min

    # Define local neighborhood structure
    struct = generate_binary_structure(2, 1)
    neighbourhood = iterate_structure(struct, neighbourhood_size)

    # Find local maxima
    local_max = maximum_filter(threshold_spectrogram, footprint=neighbourhood) == threshold_spectrogram

    # Apply erosion to remove points on the edges
    background = (threshold_spectrogram == amp_min)
    eroded_background = binary_erosion(background, structure=neighbourhood, border_value=1)
    detected_peaks = local_max & ~eroded_background

    # Extract peak positions
    peak_positions = np.argwhere(detected_peaks)

    # Get frequency and time indices
    freq_indices = peak_positions[:, 0]
    time_indices = peak_positions[:, 1]

    # Convert indices to values
    peak_freqs = [freqs[i] for i in freq_indices]
    peak_times = [times[i] for i in time_indices]

    # Create peaks list sorted by time
    peaks = list(zip(peak_freqs, peak_times))
    peaks.sort(key=lambda x: x[1])

    return peaks


def create_fingerprints(peaks, fan_out=15, time_delta_max=200):

    fingerprints = []

    for i, anchor in enumerate(peaks):
        for j in range(1, fan_out + 1):
            if i + j < len(peaks):
                point = peaks[i + j]
                freq1 = anchor[0]
                freq2 = point[0]
                t1 = anchor[1]
                t2 = point[1]
                time_delta = t2 - t1

                if 0 < time_delta <= time_delta_max:
                    hash_value = hash((freq1, freq2, time_delta))
                    fingerprints.append((hash_value, t1))

    return fingerprints


def add_song_to_database(database, song_mapping, song_id, song_name, audio_path, fingerprints):

    # Store song metadata
    song_mapping[song_id] = {"path": audio_path, "name": song_name}

    # Add fingerprints to the database
    for hash_value, time_offset in fingerprints:
        if hash_value not in database:
            database[hash_value] = []
        database[hash_value].append((song_id, time_offset))

    return database, song_mapping


def match_sample(database, song_mapping, fingerprints_sample, threshold=0.0001):

    # Match against the database
    matches = defaultdict(Counter)
    matched_count = 0

    for hash_value, sample_time in fingerprints_sample:
        if hash_value in database:
            matched_count += 1
            for song_id, song_time in database[hash_value]:
                # The time offset is how far the sample is from the start of the song
                time_offset = song_time - sample_time
                matches[song_id][time_offset] += 1

    # Find the song with the most consistent time offsets
    results = []
    for song_id, time_offsets in matches.items():
        # Skip if no matches
        if not time_offsets:
            continue

        # Find the most common time offset
        best_offset, best_count = time_offsets.most_common(1)[0]

        # Calculate a score
        score = best_count / len(fingerprints_sample) if fingerprints_sample else 0

        # Add to results if score is above threshold
        if score >= threshold:
            results.append((song_id, score, best_offset))

    # Sort results by score
    results.sort(key=lambda x: x[1], reverse=True)

    return results, matched_count


def visualize_spectrogram_with_peaks(spectrogram, sr=22050, peaks=None, title="Spectrogram"):

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format='%2.0f dB')

    if peaks:
        peak_freqs = [p[0] for p in peaks]
        peak_times = [p[1] for p in peaks]
        plt.scatter(peak_times, peak_freqs, color='blue', s=10)

    plt.title(title)
    plt.tight_layout()
    plt.show()

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

    def add_song(self, song_path, song_name=None):

        if song_name is None:
            song_name = os.path.basename(song_path)

        # Generate a new song ID
        song_id = max(self.song_mapping.keys(), default=0) + 1

        # Process the song
        spectrogram, freqs, times = load_and_process_audio(song_path)
        ## visualize_spectrogram_with_peaks(spectrogram)
        peaks = find_peaks(spectrogram, freqs, times)
        fingerprints = create_fingerprints(peaks)

        # Add to database
        self.database, self.song_mapping = add_song_to_database(
            self.database, self.song_mapping, song_id, song_name, song_path, fingerprints
        )

        print(f"Added song {song_id}: {song_name} to the database")
        print(f"Database now contains {len(self.database)} unique hash values")

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

    def identify_sample(self, sample_path, threshold=0.0001):

        spectrogram, freqs, times = load_and_process_audio(sample_path)
        peaks = find_peaks(spectrogram, freqs, times)
        fingerprints = create_fingerprints(peaks)

        # Match against the database
        results, matched_count = match_sample(self.database, self.song_mapping, fingerprints, threshold)

        print(f"Matched {matched_count} fingerprints from the sample against the database")

        # Print the results
        if results:
            print("Matches found are:")
            for song_id, score, offset in results:
                song_name = self.song_mapping[song_id]["name"]
                print(f"Song: {song_name}, Score: {score:.4f}, Time Offset: {offset:.4f}")
        else:
            print("No matches found")
        for song_id, score, offset in results:
                song_name = self.song_mapping[song_id]["name"]
                results.sort(key=lambda x: x[1])
                print(f"The Song is: {song_name}, Score: {score:.4f}, Time Offset: {offset:.4f}")
                st.text("The song is: ")
                st.text(song_name)
                break
        
        return results

    def save_database(self, database_path="fingerprint_database.pkl", mapping_path="song_mapping.pkl"):

        with open(database_path, "wb") as f:
            pickle.dump(self.database, f)

        with open(mapping_path, "wb") as f:
            pickle.dump(self.song_mapping, f)

        print(f"Database saved to {database_path} and {mapping_path}")


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
    clean_link = clean_youtube_url(video_link)
    submitted = st.form_submit_button("Upload Song")
    if submitted and clean_link:
        download_best_audio_as_mp3(clean_link, DOWNLOADS_DIR)
        raw_title = get_video_title(clean_link, DOWNLOADS_DIR)
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
