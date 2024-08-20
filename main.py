import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy import signal
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="AI Sound Design Suite", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stTabs {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stSlider>div>div>div {
        background-color: #4CAF50;
    }
    .stSelectbox>div>div {
        background-color: #262730;
        color: #FAFAFA;
    }
    .css-145kmo2 {
        color: #FAFAFA;
    }
    .css-1d391kg {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Sound generation functions (same as before)
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
    release_samples = int(release * total_samples)
    
    envelope = np.zeros_like(audio)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, 0.5, decay_samples)
    envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = 0.5
    envelope[attack_samples+decay_samples+sustain_samples:] = np.linspace(0.5, 0, release_samples)
    
    return audio * envelope

# Plot waveform with dark theme
def plot_waveform(audio, sample_rate):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(0, len(audio)) / sample_rate
    ax.plot(time, audio, color='#4CAF50')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')
    ax.grid(True, color='#555555')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#262730')
    return fig

# Main app
st.title("AI-Powered Sound Design Suite")

# Tabs
tab1, tab2 = st.tabs(["Sound Generator", "About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Waveform Settings")
        waveform = st.selectbox("Select Waveform", ["Sine", "Square", "Sawtooth"])
        frequency = st.slider("Frequency (Hz)", 20, 2000, 440)
        duration = st.slider("Duration (seconds)", 0.1, 5.0, 1.0)

        st.subheader("Envelope Settings")
        attack = st.slider("Attack", 0.0, 1.0, 0.1)
        decay = st.slider("Decay", 0.0, 1.0, 0.1)
        sustain = st.slider("Sustain", 0.0, 1.0, 0.5)
        release = st.slider("Release", 0.0, 1.0, 0.3)

    with col2:
        st.subheader("Generate Sound")
        if st.button("Generate", key="generate_button"):
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

with tab2:
    st.header("About this Sound Design Suite")
    st.write("""
    This AI-powered Sound Design Suite is a demonstration of basic sound synthesis techniques.
    It allows you to generate different waveforms, adjust their parameters, and apply an ADSR envelope.
    
    Features:
    - Generate Sine, Square, and Sawtooth waves
    - Adjust frequency and duration
    - Apply ADSR envelope
    - Visualize the generated waveform
    - Download the created sound as a WAV file
    
    Note: This is a basic demonstration. A full AI-powered sound design suite would include more advanced
    features and possibly integration with machine learning models for sound generation and manipulation.
    """)

    st.subheader("Future Enhancements")
    st.write("""
    - More complex sound synthesis techniques
    - Effects processing (reverb, delay, distortion, etc.)
    - Integration with machine learning models for AI-generated sounds
    - Multi-track mixing capabilities
    - Spectral analysis and visualization
    - MIDI input/output support
    """)

st.sidebar.title("Sound Design Suite")
st.sidebar.info("Use the controls in the main panel to generate and manipulate sounds.")
st.sidebar.warning("This is a demo version with limited features.")
