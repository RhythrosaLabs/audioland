import streamlit as st
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(page_title="Simple Sound Design Suite", layout="wide")

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

def plot_waveform(audio, sample_rate):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio) / sample_rate, len(audio)), audio)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.ylim(-1, 1)
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    return buf

# Main app
st.title("Simple Sound Design Suite")

st.header("Waveform Generator")

waveform = st.selectbox("Select Waveform", ["Sine", "Square", "Sawtooth"])
frequency = st.slider("Frequency (Hz)", 20, 2000, 440)
duration = st.slider("Duration (seconds)", 0.1, 5.0, 1.0)

st.subheader("Envelope Settings")
attack = st.slider("Attack", 0.0, 1.0, 0.1)
decay = st.slider("Decay", 0.0, 1.0, 0.1)
sustain = st.slider("Sustain", 0.0, 1.0, 0.5)
release = st.slider("Release", 0.0, 1.0, 0.3)

if st.button("Generate Waveform"):
    if waveform == "Sine":
        audio = generate_sine_wave(frequency, duration)
    elif waveform == "Square":
        audio = generate_square_wave(frequency, duration)
    else:
        audio = generate_sawtooth_wave(frequency, duration)
    
    audio = apply_envelope(audio, attack, decay, sustain, release)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Plot waveform
    waveform_plot = plot_waveform(audio, 44100)
    st.image(waveform_plot)
    
    st.write("Note: Audio playback is not available in this simplified version.")

st.sidebar.title("Simple Sound Design Suite")
st.sidebar.info("Use the controls to generate and visualize different waveforms.")
st.sidebar.warning("This is a simplified version with limited features.")
