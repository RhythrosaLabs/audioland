import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy import signal
import matplotlib.pyplot as plt
from random_samples import generate_random_samples_and_sequence
from drum_loop_generator import DrumLoopGenerator
import os
import replicate
from datetime import datetime
import requests
import shutil
import librosa
import threading
import time
from scipy.signal import convolve
import mido
from mido import Message, MidiFile, MidiTrack
import random

# Add this at the beginning of your script, after your imports and before the main app code:
if 'cleanup_needed' not in st.session_state:
    st.session_state.cleanup_needed = False

if st.session_state.cleanup_needed:
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    st.session_state.cleanup_needed = False


# Set page config
st.set_page_config(page_title="AI Sound Design Suite", layout="wide")

# Custom CSS for dark theme and global transport
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stSlider>div>div>div {
        background-color: #4CAF50;
    }
    .global-transport {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #1E1E1E;
        padding: 10px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'sounds' not in st.session_state:
    st.session_state.sounds = {}
if 'replicate_api_key' not in st.session_state:
    st.session_state.replicate_api_key = ''
if 'mixer_channels' not in st.session_state:
    st.session_state.mixer_channels = [{
        'file': None,
        'volume': 1.0,
        'pan': 0.0,
        'pitch_shift': 0,
        'reverb': 0.0,
        'delay': 0.0,
        'distortion': 0.0,
        'low_pass': 22050,
        'high_pass': 20,
        'reverse': False
    } for _ in range(8)]
if 'playback' not in st.session_state:
    st.session_state.playback = {
        'audio': None,
        'sample_rate': None,
        'is_playing': False,
        'current_position': 0,
        'loop': False,
        'bpm': 120
    }

# Create working directory
WORK_DIR = "audio_files"
os.makedirs(WORK_DIR, exist_ok=True)

# Autosave function
def autosave_audio(audio, sample_rate, prefix):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{prefix}_{timestamp}.wav"
    filepath = os.path.join(WORK_DIR, filename)
    sf.write(filepath, audio, sample_rate)
    return filepath

# Sound generation functions
def generate_sine_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * frequency * t)

def generate_square_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return signal.square(2 * np.pi * frequency * t)

def generate_sawtooth_wave(frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return signal.sawtooth(2 * np.pi * frequency * t)

def apply_envelope(audio, attack, decay, sustain, release):
    total_samples = len(audio)
    attack_samples = int(attack * total_samples)
    decay_samples = int(decay * total_samples)
    sustain_samples = int(sustain * total_samples)
    release_samples = total_samples - attack_samples - decay_samples - sustain_samples
    
    envelope = np.ones_like(audio)
    
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, 0.5, decay_samples)
    envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = 0.5
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(0.5, 0, release_samples)
    
    return audio * envelope

# Plot waveform function
def plot_waveform(audio, sample_rate):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio) / sample_rate, len(audio)), audio)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    return plt

# Session state functions
def save_sound(name, audio, sample_rate):
    st.session_state.sounds[name] = {'audio': audio, 'sample_rate': sample_rate}

def load_sound(name):
    return st.session_state.sounds[name]['audio'], st.session_state.sounds[name]['sample_rate']

def list_saved_sounds():
    return list(st.session_state.sounds.keys())

# AI Music Generator functions
def load_api_key():
    return st.session_state.replicate_api_key

def save_api_key(api_key):
    st.session_state.replicate_api_key = api_key
    os.environ["REPLICATE_API_TOKEN"] = api_key

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    else:
        st.error(f"Failed to download file: {response.status_code}")
        return False

def generate_music(input_text, duration, model):
    api_key = load_api_key()
    if not api_key:
        st.error("Please enter your Replicate API key in the sidebar and click 'Save API Key'.")
        return

    # Ensure the API key is set in the environment
    os.environ["REPLICATE_API_TOKEN"] = api_key

    model_input = {
        "prompt": input_text,
        "duration": duration
    }

    if model == 'Meta MusicGen':
        model_id = "meta/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906"
    else:  # Assume it's Loop Test model
        model_id = "allenhung1025/looptest:0de4a5f14b9120ce02c590eb9cf6c94841569fafbc4be7ab37436ce738bcf49f"

    try:
        with st.spinner("Generating music... This may take a while."):
            # Create a new Replicate client with the API key
            client = replicate.Client(api_token=api_key)
            output = client.run(model_id, input=model_input)
            
            if isinstance(output, list) and len(output) > 0:
                download_url = output[0]
            else:
                download_url = output
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            sanitized_input_text = "".join(e for e in input_text if e.isalnum())
            filename = f"{sanitized_input_text[:30]}_{timestamp}.wav"
            
            if download_file(download_url, filename):
                st.success(f"Music generated and saved as {filename}")
                
                # Load and play the generated audio
                audio_file = open(filename, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            else:
                st.error("Failed to download the generated audio file.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check that your API key is correct and that you have the necessary permissions.")

def load_audio_file(file):
    audio, sample_rate = librosa.load(file, sr=None, mono=False)
    return audio, sample_rate

def apply_volume(audio, volume):
    return audio * volume

def apply_pan(audio, pan):
    if audio.ndim == 1:
        audio = np.column_stack((audio, audio))
    left = audio[:, 0] * (1 - pan)
    right = audio[:, 1] * (1 + pan)
    return np.column_stack((left, right))

def apply_pitch_shift(audio, sample_rate, semitones):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)

def apply_reverb(audio, sample_rate, room_size):
    reverb_effect = (np.random.random(int(room_size * sample_rate)) * 2 - 1) * np.exp(-np.arange(int(room_size * sample_rate)) / (room_size * sample_rate))
    return convolve(audio, reverb_effect, mode='same')

def apply_delay(audio, sample_rate, delay_time, feedback):
    delay_samples = int(delay_time * sample_rate)
    delayed = np.zeros_like(audio)
    delayed[delay_samples:] = audio[:-delay_samples]
    return audio + feedback * delayed

def apply_distortion(audio, amount):
    return np.tanh(amount * audio)

def apply_low_pass(audio, sample_rate, cutoff):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, audio)

def apply_high_pass(audio, sample_rate, cutoff):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(6, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, audio)

def apply_reverse(audio):
    return audio[::-1]

def mix_audio(channels):
    max_length = max(len(ch['audio']) for ch in channels if ch['audio'] is not None)
    mixed_audio = np.zeros((max_length, 2))
    
    for ch in channels:
        if ch['audio'] is not None:
            audio = ch['audio']
            sample_rate = ch['sample_rate']

            # Apply effects
            audio = apply_pitch_shift(audio, sample_rate, ch['pitch_shift'])
            audio = apply_reverb(audio, sample_rate, ch['reverb'])
            audio = apply_delay(audio, sample_rate, ch['delay'], 0.5)
            audio = apply_distortion(audio, ch['distortion'])
            audio = apply_low_pass(audio, sample_rate, ch['low_pass'])
            audio = apply_high_pass(audio, sample_rate, ch['high_pass'])
            if ch['reverse']:
                audio = apply_reverse(audio)

            if audio.ndim == 1:
                audio = np.column_stack((audio, audio))
            audio = apply_volume(audio, ch['volume'])
            audio = apply_pan(audio, ch['pan'])
            
            # Pad shorter audio with zeros
            if len(audio) < max_length:
                audio = np.pad(audio, ((0, max_length - len(audio)), (0, 0)))
            
            mixed_audio += audio
    
    # Normalize the mixed audio
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
    return mixed_audio

# Transport functions
def play_audio():
    if st.session_state.playback['audio'] is not None:
        st.session_state.playback['is_playing'] = True
        # Instead of playing audio, we'll just update the state
        # The audio will be played using Streamlit's audio component

def pause_audio():
    st.session_state.playback['is_playing'] = False

def stop_audio():
    st.session_state.playback['is_playing'] = False
    st.session_state.playback['current_position'] = 0

def update_position():
    while st.session_state.playback['is_playing']:
        st.session_state.playback['current_position'] += 0.1
        if st.session_state.playback['current_position'] >= len(st.session_state.playback['audio']) / st.session_state.playback['sample_rate']:
            if st.session_state.playback['loop']:
                st.session_state.playback['current_position'] = 0
            else:
                stop_audio()
        time.sleep(0.1)
        st.experimental_rerun()  # This will update the UI

# Add this function for MIDI generation
def generate_random_midi(num_notes=50, ticks_per_beat=480):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    track.append(Message('program_change', program=random.randint(0, 127), time=0))
    
    for i in range(num_notes):
        note = random.randint(60, 84)  # C4 to C6
        velocity = random.randint(64, 127)  # medium to loud
        duration = random.randint(ticks_per_beat // 4, ticks_per_beat * 2)  # 1/16 note to 2 beats
        
        # Note on
        track.append(Message('note_on', note=note, velocity=velocity, time=random.randint(0, ticks_per_beat // 2) if i > 0 else 0))
        
        # Note off
        track.append(Message('note_off', note=note, velocity=0, time=duration))
    
    return mid


# Global transport UI
def render_global_transport():
    st.markdown('<div class="global-transport">', unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 2, 1])
    with cols[0]:
        if st.button("Play"):
            play_audio()
            threading.Thread(target=update_position, daemon=True).start()
    with cols[1]:
        if st.button("Pause"):
            pause_audio()
    with cols[2]:
        if st.button("Stop"):
            stop_audio()
    with cols[3]:
        if st.session_state.playback['audio'] is not None:
            duration = len(st.session_state.playback['audio']) / st.session_state.playback['sample_rate']
            st.session_state.playback['current_position'] = st.slider("", 0.0, duration, st.session_state.playback['current_position'], key="global_seek")
    with cols[4]:
        st.session_state.playback['loop'] = st.checkbox("Loop", value=st.session_state.playback['loop'], key="global_loop")
    st.markdown('</div>', unsafe_allow_html=True)

# Main app
st.title("AI-Powered Sound Design Suite")

# Sidebar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Replicate API key:", type="password")
if st.sidebar.button("Save API Key"):
    save_api_key(api_key)
    st.sidebar.success("API key saved!")


# Modify the tabs section to include the new MIDI Generator tab
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Sound Generator", "Sound Library", "Random Samples", "Drum Loop Generator", "AI Music Generator", "Mixer", "MIDI Generator"])

with tab1:
    st.header("Waveform Generator")

    waveform = st.selectbox("Select Waveform", ["Sine", "Square", "Sawtooth"])
    frequency = st.slider("Frequency (Hz)", 20, 2000, 440)
    duration = st.slider("Duration (seconds)", 0.1, 5.0, 1.0)

    st.subheader("Envelope Settings")
    attack = st.slider("Attack", 0.0, 1.0, 0.1)
    decay = st.slider("Decay", 0.0, 1.0, 0.1)
    sustain = st.slider("Sustain", 0.0, 1.0, 0.5)
    release = st.slider("Release", 0.0, 1.0, 0.3)

    if st.button("Generate Sound"):
        if waveform == "Sine":
            audio = generate_sine_wave(frequency, duration)
        elif waveform == "Square":
            audio = generate_square_wave(frequency, duration)
        else:
            audio = generate_sawtooth_wave(frequency, duration)
        
        audio = apply_envelope(audio, attack, decay, sustain, release)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Save audio to a BytesIO object
        buffer = io.BytesIO()
        sf.write(buffer, audio, 44100, format='WAV')
        buffer.seek(0)
        
        # Display audio player
        st.audio(buffer, format='audio/wav')
        
        # Provide download link
        st.download_button(
            label="Download WAV",
            data=buffer,
            file_name="generated_sound.wav",
            mime="audio/wav"
        )

        # Plot waveform
        st.pyplot(plot_waveform(audio, 44100))

        # Save to session state option
        save_name = st.text_input("Save sound as:")
        if st.button("Save Sound"):
            save_sound(save_name, audio, 44100)
            st.success(f"Saved {save_name} to library!")

with tab2:
    st.header("Sound Library")
    
    saved_sounds = list_saved_sounds()
    if saved_sounds:
        selected_sound = st.selectbox("Select a saved sound", saved_sounds)
        if st.button("Load Sound"):
            audio, sample_rate = load_sound(selected_sound)
            
            # Create a buffer for the loaded sound
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)
            
            # Display audio player for the loaded sound
            st.audio(buffer, format='audio/wav')
            
            # Plot waveform of the loaded sound
            st.pyplot(plot_waveform(audio, sample_rate))
    else:
        st.write("No saved sounds found in the library.")

with tab3:
    st.header("Random Samples Generator")
    
    if st.button("Generate Random Samples"):
        with st.spinner("Generating random samples..."):
            samples, sequence = generate_random_samples_and_sequence()
        
        st.success("Random samples generated successfully!")
        
        # Display individual samples
        st.subheader("Individual Samples")
        for i, sample in enumerate(samples):
            st.audio(sample, format='audio/wav')
            st.download_button(
                label=f"Download Sample {i+1}",
                data=sample,
                file_name=f"random_sample_{i+1}.wav",
                mime="audio/wav"
            )
        
        # Display musical sequence
        st.subheader("Musical Sequence")
        st.audio(sequence, format='audio/wav')
        st.download_button(
            label="Download Musical Sequence",
            data=sequence,
            file_name="musical_sequence.wav",
            mime="audio/wav"
        )

with tab4:
    st.header("Drum Loop Generator")
    
    tempo = st.slider("Tempo (BPM)", 60, 200, 120)
    beat_length = st.slider("Beat Length", 4, 32, 16)
    
    if st.button("Generate Drum Loop"):
        with st.spinner("Generating drum loop..."):
            generator = DrumLoopGenerator(tempo=tempo, beat_length=beat_length)
            drum_loop = generator.generate_loop()
        
        st.success("Drum loop generated successfully!")
        
        # Display drum loop
        st.audio(drum_loop, format='audio/wav')
        st.download_button(
            label="Download Drum Loop",
            data=drum_loop,
            file_name="drum_loop.wav",
            mime="audio/wav"
        )

with tab5:
    st.header("AI Music Generator")
    
    st.warning("Make sure you've entered and saved your Replicate API key in the sidebar before generating music.")
    
    input_text = st.text_area("Enter a prompt for the music:", "An upbeat electronic dance track with a catchy melody")
    duration = st.slider("Duration (seconds)", 5, 60, 30)
    model = st.selectbox("Select AI Model", ["Meta MusicGen", "Loop Test"])
    
    if st.button("Generate Music"):
        generate_music(input_text, duration, model)

with tab6:
    st.header("8-Channel Mixer with Effects")
    
    # BPM setting
    st.session_state.playback['bpm'] = st.number_input("BPM", min_value=1, max_value=300, value=st.session_state.playback['bpm'])
    
    # Channel controls
    for i in range(8):
        st.subheader(f"Channel {i+1}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uploaded_file = st.file_uploader(f"Import audio for channel {i+1}", type=['wav', 'mp3'])
            if uploaded_file is not None:
                file_contents = uploaded_file.read()
                with open(os.path.join(WORK_DIR, uploaded_file.name), 'wb') as f:
                    f.write(file_contents)
                st.session_state.mixer_channels[i]['file'] = os.path.join(WORK_DIR, uploaded_file.name)
                st.success(f"File uploaded and saved: {uploaded_file.name}")
        
        with col2:
            st.session_state.mixer_channels[i]['volume'] = st.slider(f"Volume {i+1}", 0.0, 1.0, 1.0, key=f"vol_{i}")
        
        with col3:
            st.session_state.mixer_channels[i]['pan'] = st.slider(f"Pan {i+1}", -1.0, 1.0, 0.0, key=f"pan_{i}")
        
        # Effects
        st.session_state.mixer_channels[i]['pitch_shift'] = st.slider(f"Pitch Shift {i+1} (semitones)", -12, 12, 0, key=f"pitch_{i}")
        st.session_state.mixer_channels[i]['reverb'] = st.slider(f"Reverb {i+1}", 0.0, 1.0, 0.0, key=f"reverb_{i}")
        st.session_state.mixer_channels[i]['delay'] = st.slider(f"Delay {i+1} (seconds)", 0.0, 1.0, 0.0, key=f"delay_{i}")
        st.session_state.mixer_channels[i]['distortion'] = st.slider(f"Distortion {i+1}", 0.0, 1.0, 0.0, key=f"dist_{i}")
        st.session_state.mixer_channels[i]['low_pass'] = st.slider(f"Low Pass {i+1} (Hz)", 20, 22050, 22050, key=f"lp_{i}")
        st.session_state.mixer_channels[i]['high_pass'] = st.slider(f"High Pass {i+1} (Hz)", 20, 22050, 20, key=f"hp_{i}")
        st.session_state.mixer_channels[i]['reverse'] = st.checkbox(f"Reverse {i+1}", key=f"rev_{i}")
    
    if st.button("Mix Audio"):
        mixed_channels = []
        for ch in st.session_state.mixer_channels:
            if ch['file'] is not None:
                audio, sample_rate = load_audio_file(ch['file'])
                mixed_channels.append({
                    'audio': audio,
                    'sample_rate': sample_rate,
                    'volume': ch['volume'],
                    'pan': ch['pan'],
                    'pitch_shift': ch['pitch_shift'],
                    'reverb': ch['reverb'],
                    'delay': ch['delay'],
                    'distortion': ch['distortion'],
                    'low_pass': ch['low_pass'],
                    'high_pass': ch['high_pass'],
                    'reverse': ch['reverse']
                })
        
        if mixed_channels:
            mixed_audio = mix_audio(mixed_channels)
            mixed_filepath = autosave_audio(mixed_audio, sample_rate, "mixed_audio")
            st.success(f"Mixed audio saved: {mixed_filepath}")
            
            # Update playback state
            st.session_state.playback['audio'] = mixed_audio
            st.session_state.playback['sample_rate'] = sample_rate
            st.session_state.playback['current_position'] = 0
            
            # Display waveform
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(mixed_audio)
            ax.set_xlabel('Samples')
            ax.set_ylabel('Amplitude')
            ax.set_title('Mixed Audio Waveform')
            st.pyplot(fig)
            
            # Create a buffer for the mixed audio
            buffer = io.BytesIO()
            sf.write(buffer, mixed_audio, sample_rate, format='WAV')
            buffer.seek(0)
            
            # Display audio player for the mixed audio
            st.audio(buffer, format='audio/wav')
        else:
            st.warning("No audio files loaded in the mixer channels.")


# Add this new tab for MIDI Generator
with tab7:
    st.header("Random MIDI Generator")
    
    num_notes = st.slider("Number of Notes", 10, 100, 50)
    
    if st.button("Generate Random MIDI"):
        with st.spinner("Generating random MIDI sequence..."):
            midi_data = generate_random_midi(num_notes)
            
            # Save MIDI file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            midi_filename = f"random_midi_{timestamp}.mid"
            midi_filepath = os.path.join(WORK_DIR, midi_filename)
            midi_data.save(midi_filepath)
            
            st.success(f"Random MIDI sequence generated: {midi_filename}")
            
            # Provide download link
            with open(midi_filepath, "rb") as f:
                st.download_button(
                    label="Download MIDI File",
                    data=f,
                    file_name=midi_filename,
                    mime="audio/midi"
                )
            
            # Display MIDI information
            st.subheader("MIDI Sequence Information")
            st.write(f"Number of tracks: {len(midi_data.tracks)}")
            st.write(f"Number of notes: {num_notes}")
            
            # Display first few notes
            st.subheader("First 10 Notes")
            notes = []
            for i, msg in enumerate(midi_data.tracks[0]):
                if msg.type == 'note_on' and len(notes) < 10:
                    notes.append(f"Note: {msg.note}, Velocity: {msg.velocity}")
            for note in notes:
                st.write(note)


# Render global transport
render_global_transport()

st.sidebar.title("Sound Design Suite")
st.sidebar.info("Use the tabs to switch between different sound design tools.")

# Cleanup function
def cleanup_work_dir():
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)

# Register the cleanup function to run when the app is closed
st.on_script_run_end(cleanup_work_dir)
