import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy import signal
import matplotlib.pyplot as plt
from random_samples import generate_random_samples_and_sequence  # Import the new function

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

# Initialize session state for storing sounds
if 'sounds' not in st.session_state:
    st.session_state.sounds = {}

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

# Main app
st.title("AI-Powered Sound Design Suite")

# Tabs
tab1, tab2, tab3 = st.tabs(["Sound Generator", "Sound Library", "Random Samples"])

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
