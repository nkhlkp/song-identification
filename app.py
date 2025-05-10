import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import mlab
from scipy.ndimage import maximum_filter, generate_binary_structure, iterate_structure, binary_erosion
import matplotlib.pyplot as plt
import librosa
import os
from collections import defaultdict, Counter
import pickle
import tempfile
from pydub import AudioSegment

# Create necessary directories
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

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
                song_name = self.song_mapping[song_id]["name"] if isinstance(self.song_mapping[song_id], dict) else self.song_mapping[song_id]
                print(f"Song: {song_name}, Score: {score:.4f}, Time Offset: {offset:.4f}")
        else:
            print("No matches found")
            
        return results

# ---------------------------------------------------------------------------------------------

st.title("Song Identification App")

# Initialize or load database
if os.path.exists("fingerprint_database.pkl") and os.path.exists("song_mapping.pkl"):
    fingerprinter = AudioFingerprinter("fingerprint_database.pkl", "song_mapping.pkl")
    st.success("Fingerprint database loaded successfully!")
else:
    fingerprinter = AudioFingerprinter()
    st.warning("No fingerprint database found. Please ensure the database files are present.")

# Create tabs for different identification methods
tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

# Record audio form
with tab1:
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
                for i, (song_id, score, offset) in enumerate(results, 1):
                    song_name = fingerprinter.song_mapping[song_id]["name"] if isinstance(fingerprinter.song_mapping[song_id], dict) else fingerprinter.song_mapping[song_id]
                    st.write(f"{i}. {song_name} (Confidence: {score:.2%})")
            else:
                st.warning("No matches found in the database.")

# Upload audio file form
with tab2:
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
                for i, (song_id, score, offset) in enumerate(results, 1):
                    song_name = fingerprinter.song_mapping[song_id]["name"] if isinstance(fingerprinter.song_mapping[song_id], dict) else fingerprinter.song_mapping[song_id]
                    st.write(f"{i}. {song_name} (Confidence: {score:.2%})")
            else:
                st.warning("No matches found in the database.")

# Display songs in database
st.header("Songs in Database")

if st.button("Show all songs in database"):
    # Check if the database exists and has songs
    if hasattr(fingerprinter, 'song_mapping') and fingerprinter.song_mapping:
        songs_list = []
        # Build list with custom index and song name
        for i, (song_id, song_data) in enumerate(fingerprinter.song_mapping.items(), start=1):
            try:
                song_name = song_data.get("name", "Unknown") if isinstance(song_data, dict) else song_data
            except Exception as e:
                song_name = "Invalid format"
            songs_list.append({"Index": i, "Song Name": song_name})
        
        # Create DataFrame and reset index to hide default one
        songs_df = pd.DataFrame(songs_list)
        songs_df = songs_df.reset_index(drop=True)
        
        # Display clean table without extra index
        st.dataframe(songs_df, use_container_width=True)
        
        # Show total number of songs
        st.write(f"Total songs in database: {len(fingerprinter.song_mapping)}")
    else:
        st.info("No songs found in the database.")
