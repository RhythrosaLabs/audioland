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

# Set page config
st.set_page_config(page_title="AI Sound Design Suite", layout="wide")

# Custom CSS for dark theme
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
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing sounds and API key
if 'sounds' not in st.session_state:
    st.session_state.sounds = {}
if 'replicate_api_key' not in st.session_state:
    st.session_state.replicate_api_key = ''


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

# New functions for generative AI audio
def load_api_key():
    return st.session_state.replicate_api_key

def save_api_key(api_key):
    st.session_state.replicate_api_key = api_key

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
        st.error("Please enter your Replicate API key in the sidebar.")
        return

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
            output = replicate.run(model_id, input=model_input)
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

# Main app
st.title("AI-Powered Sound Design Suite")

# Sidebar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Replicate API key:", type="password")
if st.sidebar.button("Save API Key"):
    save_api_key(api_key)
    st.sidebar.success("API key saved!")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sound Generator", "Sound Library", "Random Samples", "Drum Loop Generator", "AI Music Generator"])


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
    
    input_text = st.text_area("Enter a prompt for the music:", "An upbeat electronic dance track with a catchy melody")
    duration = st.slider("Duration (seconds)", 5, 60, 30)
    model = st.selectbox("Select AI Model", ["Meta MusicGen", "Loop Test"])
    
    if st.button("Generate Music"):
        generate_music(input_text, duration, model)

st.sidebar.title("Sound Design Suite")
st.sidebar.info("Use the tabs to switch between different sound design tools.")
