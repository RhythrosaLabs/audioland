import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy import signal
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range

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

# Initialize session state
if 'library' not in st.session_state:
    st.session_state.library = {}
if 'mixer_tracks' not in st.session_state:
    st.session_state.mixer_tracks = []
if 'sequence' not in st.session_state:
    st.session_state.sequence = []

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

# Effects functions
def apply_delay(audio, delay_time, feedback, mix, sample_rate=44100):
    delay_samples = int(delay_time * sample_rate)
    delayed = np.zeros_like(audio)
    delayed[delay_samples:] = audio[:-delay_samples]
    output = audio + feedback * delayed
    return (1 - mix) * audio + mix * output

def apply_reverb(audio, room_size, damping, sample_rate=44100):
    # This is a simple reverb approximation
    impulse_response = np.exp(-damping * np.arange(int(room_size * sample_rate))) * np.random.randn(int(room_size * sample_rate))
    return signal.convolve(audio, impulse_response, mode='same')

# Function to save audio to library
def save_to_library(name, audio, sample_rate):
    st.session_state.library[name] = {'audio': audio, 'sample_rate': sample_rate}
    
# Function to load audio from library
def load_from_library(name):
    return st.session_state.library[name]['audio'], st.session_state.library[name]['sample_rate']

# Plot waveform function
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
tab1, tab2, tab3, tab4 = st.tabs(["Sound Generator", "Library", "Mixer", "Sequencer"])

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

        st.subheader("Effects")
        delay_time = st.slider("Delay Time (s)", 0.0, 1.0, 0.0)
        delay_feedback = st.slider("Delay Feedback", 0.0, 1.0, 0.0)
        delay_mix = st.slider("Delay Mix", 0.0, 1.0, 0.0)
        reverb_size = st.slider("Reverb Size", 0.0, 1.0, 0.0)
        reverb_damping = st.slider("Reverb Damping", 0.0, 1.0, 0.5)

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
            
            if delay_time > 0:
                audio = apply_delay(audio, delay_time, delay_feedback, delay_mix)
            
            if reverb_size > 0:
                audio = apply_reverb(audio, reverb_size, reverb_damping)
            
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

            # Save to library option
            save_name = st.text_input("Save to library as:")
            if st.button("Save to Library"):
                save_to_library(save_name, audio, 44100)
                st.success(f"Saved {save_name} to library!")

with tab2:
    st.header("Sound Library")
    
    # Display and allow playback of saved sounds
    for name, data in st.session_state.library.items():
        st.subheader(name)
        buffer = io.BytesIO()
        sf.write(buffer, data['audio'], data['sample_rate'], format='WAV')
        buffer.seek(0)
        st.audio(buffer, format='audio/wav')
        
        if st.button(f"Add {name} to Mixer"):
            st.session_state.mixer_tracks.append({'name': name, 'audio': data['audio'], 'volume': 1.0, 'pan': 0.0})
            st.success(f"Added {name} to Mixer!")

with tab3:
    st.header("Mixer")
    
    # Display mixer tracks
    for i, track in enumerate(st.session_state.mixer_tracks):
        st.subheader(f"Track {i+1}: {track['name']}")
        track['volume'] = st.slider(f"Volume {i+1}", 0.0, 1.0, track['volume'], key=f"vol_{i}")
        track['pan'] = st.slider(f"Pan {i+1}", -1.0, 1.0, track['pan'], key=f"pan_{i}")
    
    if st.button("Mix and Export"):
        # Mix tracks
        mixed_audio = np.zeros_like(st.session_state.mixer_tracks[0]['audio'])
        for track in st.session_state.mixer_tracks:
            panned_audio = np.array([track['audio'] * (1 - track['pan']), track['audio'] * (1 + track['pan'])]).T
            mixed_audio += panned_audio * track['volume']
        
        # Normalize mixed audio
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
        # Export mixed audio
        buffer = io.BytesIO()
        sf.write(buffer, mixed_audio, 44100, format='WAV')
        buffer.seek(0)
        
        st.audio(buffer, format='audio/wav')
        st.download_button(
            label="Download Mixed Audio",
            data=buffer,
            file_name="mixed_audio.wav",
            mime="audio/wav"
        )

with tab4:
    st.header("Sequencer")
    
    # Simple sequencer
    num_steps = st.number_input("Number of steps", 1, 16, 8)
    
    # Create sequence if it doesn't exist
    if len(st.session_state.sequence) != num_steps:
        st.session_state.sequence = [None] * num_steps
    
    # Display sequence
    cols = st.columns(num_steps)
    for i in range(num_steps):
        with cols[i]:
            st.session_state.sequence[i] = st.selectbox(f"Step {i+1}", 
                                                        [None] + list(st.session_state.library.keys()), 
                                                        index=0 if st.session_state.sequence[i] is None else 
                                                        list(st.session_state.library.keys()).index(st.session_state.sequence[i]) + 1,
                                                        key=f"seq_{i}")
    
    # Sequence playback (without real-time audio)
    if st.button("Generate Sequence"):
        sequence_audio = []
        for step in st.session_state.sequence:
            if step:
                audio, _ = load_from_library(step)
                sequence_audio.append(audio)
            else:
                sequence_audio.append(np.zeros(44100))  # 1 second of silence
        
        full_sequence = np.concatenate(sequence_audio)
        
        # Normalize the sequence
        full_sequence = full_sequence / np.max(np.abs(full_sequence))
        
        # Create a buffer for the sequence
        buffer = io.BytesIO()
        sf.write(buffer, full_sequence, 44100, format='WAV')
        buffer.seek(0)
        
        # Display audio player for the sequence
        st.audio(buffer, format='audio/wav')
        
        # Provide download link for the sequence
        st.download_button(
            label="Download Sequence",
            data=buffer,
            file_name="sequence.wav",
            mime="audio/wav"
        )

st.sidebar.title("Sound Design Suite")
st.sidebar.info("Use the tabs to navigate between different features of the suite.")
st.sidebar.warning("This is a demo version with limited features.")
