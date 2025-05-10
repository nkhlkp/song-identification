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
from os import path 
from pydub import AudioSegment
import re
import ffmpeg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "fingerprint_database.pkl")
MAP_PATH = os.path.join(BASE_DIR, "song_mapping.pkl")

FFMPEG_DIR = os.path.join(os.path.dirname(__file__), "bin")
os.environ["PATH"] += os.pathsep + FFMPEG_DIR

# ---------------------------------------------------------------------------------------------


def sanitize_filename(filename):
    # Remove characters not allowed in Windows filenames
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def download_best_audio_as_mp3(video_url, save_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '0',
        }],
        'ffmpeg_location': FFMPEG_DIR,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        title = info_dict.get("title", None)
        filename = ydl.prepare_filename(info_dict)
        mp3_filename = os.path.splitext(filename)[0] + ".mp3"
        return mp3_filename, title

def get_video_title(video_url, save_path=DOWNLOAD_DIR):
    ydl_opts = {
    'outtmpl': save_path + '/%(title)s.%(ext)s',  # Save path and file name
    'postprocessors': [{  # Post-process to convert to MP3
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',  # Convert to mp3
        'preferredquality': '0',  # '0' means best quality, auto-determined by source
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
        """
        Initialize the AudioFingerprinter.

        Parameters:
        - database_path: Optional path to load an existing database
        - mapping_path: Optional path to load an existing mapping
        """
        self.database = {}
        self.song_mapping = {}

        # Load existing database if provided
        if database_path and mapping_path:
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

        # Process the sample
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
        """
        Save the database to disk.

        Parameters:
        - database_path: Path to save the database
        - mapping_path: Path to save the mapping
        """
        with open(database_path, "wb") as f:
            pickle.dump(self.database, f)

        with open(mapping_path, "wb") as f:
            pickle.dump(self.song_mapping, f)

        print(f"Database saved to {database_path} and {mapping_path}")


# ---------------------------------------------------------------------------------------------

st.title("Song Identification App")

if st.button("Load existing songs"):
    # Initialize the fingerprinter, create an object of the class
    fingerprinter = AudioFingerprinter()

    # Add all songs from the downloads directory
    fingerprinter.add_songs_from_directory(DOWNLOAD_DIR)

    # Save the database
    fingerprinter.save_database()

    # Display the message to proceed
    st.write("Great! Proceed now.")


submitted_link = None
sample_recorded_audio = None
sample_uploaded_audio = None

with st.form("get_link"):
    video_link = st.text_input("Enter the YouTube URL of the song: ")
    st.form_submit_button("Upload Song")

    if video_link:
        st.write(f"URL submitted successfully.")
    else:
        st.write(f"Please paste link before submitting.")

with st.form("get_audget_sample_from_microphone"):
    sample_recorded_audio = st.audio_input(label="Record Audio")
    submitted = st.form_submit_button("Submit")

    if sample_recorded_audio:
        st.write(f"Sample recorded and uploaded successfully.")
    else:
        st.write(f"Please do something before submitting.")

with st.form("get_sample_from_file"):
    sample_uploaded_audio = st.file_uploader(label="Upload Audio")
    submitted = st.form_submit_button("Submit")

    if sample_recorded_audio:
        st.write(f"Sample file uploaded successfully.")
    else:
        st.write(f"Please do something before submitting.")


if (video_link):
    mp3_path, video_title = download_best_audio_as_mp3(video_link, DOWNLOAD_DIR)
    video_title = sanitize_filename(get_video_title(video_link, DOWNLOAD_DIR))
    video_file_path = os.path.join(DOWNLOAD_DIR, f"{video_title}.mp3")

    st.write(f"Processing file: {video_file_path}")
    if not os.path.exists(video_file_path):
        st.error("MP3 file not found. Check the file path or name.")

    # Add the song to the fingerprint database

    fingerprinter = AudioFingerprinter(DB_PATH, MAP_PATH)

    fingerprinter.add_song(mp3_path, video_title)

    # Save the database
    fingerprinter.save_database()

    st.write("Song uploaded successfully.")

    
if sample_recorded_audio:
    with open("recorded_audio.wav", "wb") as f:
        f.write(sample_recorded_audio.getbuffer())
        st.write("Audio recorded and saved successfully!")

        # assign files 
        input_file = "recorded_audio.wav"
        output_file = "sample_recorded_audio.mp3"

        # convert wav file to mp3 file 
        sound = AudioSegment.from_wav(input_file) 
        sound.export(output_file, format="mp3")

        # Send the sample for matching
        fingerprinter = AudioFingerprinter(DB_PATH, MAP_PATH)
        results = fingerprinter.identify_sample(output_file)


if sample_uploaded_audio:

    # Send the sample for matching
    fingerprinter = AudioFingerprinter(DB_PATH, MAP_PATH)
    results = fingerprinter.identify_sample(sample_uploaded_audio)
