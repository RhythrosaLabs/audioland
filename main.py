import streamlit as st
import numpy as np
import soundfile as sf
import io
from scipy import signal
import os
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import librosa
import threading
import time
from scipy.signal import convolve
import mido
from mido import MidiFile, MidiTrack
import random
import plotly.graph_objs as go
import replicate
from drum_loop_generator import DrumLoopGenerator
from random_samples import generate_random_samples_and_sequence

# Set page config
st.set_page_config(page_title="SOUNDSTORM", layout="wide")

# Custom CSS for enhanced UI/UX
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

        .stApp {
            background-color: #0e0e0e;
            color: #ffffff;
            font-family: 'Poppins', sans-serif;
        }

        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 8px;
            padding: 10px;
            border: none;
        }

        .stButton>button:hover {
            background-color: #ff1a1a;
        }

        .stSlider > div > div > div > div {
            background: linear-gradient(to right, #ff4b4b 0%, #ff1a1a 100%);
        }

        .channel-fader {
            height: 150px;
            background-color: #ff4b4b;
            border-radius: 4px;
        }

        .channel {
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #333;
            border-radius: 10px;
            background-color: #1e1e1e;
        }

        .effects-button {
            background-color: #ff4b4b;
            border: none;
            color: white;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 8px;
        }

        .effects-button:hover {
            background-color: #ff1a1a;
        }

        .mixer-panel {
            display: flex;
            justify-content: space-between;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
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
            'reverse': False,
            'mute': False,
            'solo': False
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

init_session_state()

# Create working directory
WORK_DIR = "audio_files"
os.makedirs(WORK_DIR, exist_ok=True)

# Function to generate waveforms
def generate_waveform(waveform_type, frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    if waveform_type == 'Sine':
        return np.sin(2 * np.pi * frequency * t)
    elif waveform_type == 'Square':
        return signal.square(2 * np.pi * frequency * t)
    elif waveform_type == 'Sawtooth':
        return signal.sawtooth(2 * np.pi * frequency * t)
    elif waveform_type == 'Triangle':
        return signal.sawtooth(2 * np.pi * frequency * t, 0.5)
    elif waveform_type == 'White Noise':
        return np.random.uniform(-1, 1, t.shape)
    else:
        return np.zeros_like(t)

def apply_envelope(audio, attack, decay, sustain_level, sustain_duration, release, sample_rate=44100):
    total_samples = len(audio)
    envelope = np.zeros(total_samples)
    attack_samples = int(attack * sample_rate)
    decay_samples = int(decay * sample_rate)
    sustain_samples = int(sustain_duration * sample_rate)
    release_samples = int(release * sample_rate)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain_level, decay_samples)
    if sustain_samples > 0:
        envelope[attack_samples+decay_samples:attack_samples+decay_samples+sustain_samples] = sustain_level
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
    return audio * envelope

def plot_waveform(audio, sample_rate):
    t = np.linspace(0, len(audio) / sample_rate, num=len(audio))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=audio, mode='lines', line=dict(color='#ff4b4b')))
    fig.update_layout(template='plotly_dark', xaxis_title='Time (s)', yaxis_title='Amplitude')
    return fig

def plot_spectrum(audio, sample_rate):
    fft_spectrum = np.fft.rfft(audio)
    freq = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitude = np.abs(fft_spectrum)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=20*np.log10(magnitude), mode='lines', line=dict(color='#ff4b4b')))
    fig.update_layout(template='plotly_dark', xaxis_title='Frequency (Hz)', yaxis_title='Magnitude (dB)')
    return fig

# Main app
st.title("🎛️ AI-Powered Sound Design Suite")

# Sidebar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("🔑 Enter your Replicate API key:", type="password")
if st.sidebar.button("Save API Key"):
    st.session_state.replicate_api_key = api_key
    os.environ["REPLICATE_API_TOKEN"] = api_key
    st.sidebar.success("API key saved!")

# Tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🎵 Sound Generator", "📚 Sound Library", "🎲 Random Samples", "🥁 Drum Loop Generator", "🤖 AI Music Generator", "🎚️ Mixer", "🎹 MIDI Generator"])

with tab1:
    st.header("🎵 Advanced Waveform Generator")
    col1, col2 = st.columns(2)

    with col1:
        waveform = st.selectbox("Select Waveform", ["Sine", "Square", "Sawtooth", "Triangle", "White Noise"])
        frequency = st.slider("Frequency (Hz)", 20, 2000, 440)
        duration = st.slider("Duration (seconds)", 0.1, 5.0, 1.0)

        st.subheader("Envelope Settings")
        attack = st.slider("Attack (s)", 0.0, 1.0, 0.1)
        decay = st.slider("Decay (s)", 0.0, 1.0, 0.1)
        sustain_level = st.slider("Sustain Level", 0.0, 1.0, 0.7)
        sustain_duration = st.slider("Sustain Duration (s)", 0.0, 5.0, 0.5)
        release = st.slider("Release (s)", 0.0, 1.0, 0.3)

        if st.button("Generate Sound"):
            audio = generate_waveform(waveform, frequency, duration)
            audio = apply_envelope(audio, attack, decay, sustain_level, sustain_duration, release)
            audio = audio / np.max(np.abs(audio))

            buffer = io.BytesIO()
            sf.write(buffer, audio, 44100, format='WAV')
            buffer.seek(0)
            st.audio(buffer, format='audio/wav')
            st.download_button("Download WAV", data=buffer, file_name="generated_sound.wav", mime="audio/wav")

            st.session_state.generated_audio = audio
            st.session_state.sample_rate = 44100

    with col2:
        if 'generated_audio' in st.session_state:
            st.plotly_chart(plot_waveform(st.session_state.generated_audio, st.session_state.sample_rate), use_container_width=True)
            st.plotly_chart(plot_spectrum(st.session_state.generated_audio, st.session_state.sample_rate), use_container_width=True)
        else:
            st.info("Generate a sound to see waveform and spectrum.")

with tab2:
    st.header("📚 Sound Library")
    saved_sounds = list(st.session_state.sounds.keys())
    if saved_sounds:
        selected_sound = st.selectbox("Select a saved sound", saved_sounds)
        if st.button("Load Sound"):
            audio = st.session_state.sounds[selected_sound]['audio']
            sample_rate = st.session_state.sounds[selected_sound]['sample_rate']

            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)
            st.audio(buffer, format='audio/wav')
            st.plotly_chart(plot_waveform(audio, sample_rate), use_container_width=True)
    else:
        st.warning("No saved sounds found in the library.")

with tab3:
    st.header("🎲 Random Samples Generator")
    
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
    st.header("🥁 Drum Loop Generator")
    
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
    st.header("🤖 AI Music Generator")
    
    st.warning("Make sure you've entered and saved your Replicate API key in the sidebar before generating music.")
    
    input_text = st.text_area("Enter a prompt for the music:", "An upbeat electronic dance track with a catchy melody")
    duration = st.slider("Duration (seconds)", 5, 60, 30)
    model = st.selectbox("Select AI Model", ["Meta MusicGen", "Loop Test"])
    
    def generate_music(input_text, duration, model):
        api_key = st.session_state.replicate_api_key
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
        else:
            model_id = "allenhung1025/looptest:0de4a5f14b9120ce02c590eb9cf6c94841569fafbc4be7ab37436ce738bcf49f"

        try:
            with st.spinner("Generating music..."):
                client = replicate.Client(api_token=api_key)
                output = client.run(model_id, input=model_input)
                download_url = output[0] if isinstance(output, list) else output

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                sanitized_input_text = "".join(e for e in input_text if e.isalnum())
                filename = f"{sanitized_input_text[:30]}_{timestamp}.wav"
                
                response = requests.get(download_url)
                if response.status_code == 200:
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    st.success(f"Music generated and saved as {filename}")
                    
                    audio_file = open(filename, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/wav')
                else:
                    st.error(f"Failed to download the generated audio file. Status code: {response.status_code}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    if st.button("Generate Music"):
        generate_music(input_text, duration, model)

with tab6:
    st.header("🎚️ 8-Channel Mixer with Effects")

    st.session_state.playback['bpm'] = st.number_input("BPM", min_value=1, max_value=300, value=st.session_state.playback['bpm'])

    for i in range(8):
        with st.expander(f"Channel {i+1}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                uploaded_file = st.file_uploader(f"Upload audio for Channel {i+1}", type=['wav', 'mp3'], key=f'file_{i}')
                if uploaded_file:
                    file_path = os.path.join(WORK_DIR, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.mixer_channels[i]['file'] = file_path
                    st.success(f"Loaded {uploaded_file.name}")
            with col2:
                st.session_state.mixer_channels[i]['mute'] = st.checkbox("Mute", key=f"mute_{i}")
                st.session_state.mixer_channels[i]['solo'] = st.checkbox("Solo", key=f"solo_{i}")

            st.session_state.mixer_channels[i]['volume'] = st.slider(f"Volume {i+1}", 0.0, 1.0, st.session_state.mixer_channels[i]['volume'], key=f"vol_{i}")
            st.session_state.mixer_channels[i]['pan'] = st.slider(f"Pan {i+1}", -1.0, 1.0, st.session_state.mixer_channels[i]['pan'], key=f"pan_{i}")

            with st.expander("Effects"):
                st.session_state.mixer_channels[i]['pitch_shift'] = st.slider(f"Pitch Shift {i+1} (semitones)", -12, 12, st.session_state.mixer_channels[i]['pitch_shift'], key=f"pitch_{i}")
                st.session_state.mixer_channels[i]['reverb'] = st.slider(f"Reverb {i+1}", 0.0, 1.0, st.session_state.mixer_channels[i]['reverb'], key=f"reverb_{i}")
                st.session_state.mixer_channels[i]['delay'] = st.slider(f"Delay {i+1} (seconds)", 0.0, 1.0, st.session_state.mixer_channels[i]['delay'], key=f"delay_{i}")
                st.session_state.mixer_channels[i]['distortion'] = st.slider(f"Distortion {i+1}", 0.0, 1.0, st.session_state.mixer_channels[i]['distortion'], key=f"dist_{i}")
                st.session_state.mixer_channels[i]['low_pass'] = st.slider(f"Low Pass {i+1} (Hz)", 20, 22050, st.session_state.mixer_channels[i]['low_pass'], key=f"lp_{i}")
                st.session_state.mixer_channels[i]['high_pass'] = st.slider(f"High Pass {i+1} (Hz)", 20, 22050, st.session_state.mixer_channels[i]['high_pass'], key=f"hp_{i}")

    if st.button("Mix Audio"):
        st.info("Mixing audio...")

with tab7:
    st.header("🎹 Random MIDI Generator")
    
    num_notes = st.slider("Number of Notes", 10, 100, 50)
    
    if st.button("Generate Random MIDI"):
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        track.append(mido.Message('program_change', program=random.randint(0, 127), time=0))
        
        for i in range(num_notes):
            note = random.randint(60, 84)
            velocity = random.randint(64, 127)
            duration = random.randint(240, 960)
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=0))
            track.append(mido.Message('note_off', note=note, velocity=0, time=duration))
        
        midi_bytes = io.BytesIO()
        mid.save(file=midi_bytes)
        midi_bytes.seek(0)
        
        st.download_button("Download MIDI", data=midi_bytes.getvalue(), file_name="random_midi.mid", mime="audio/midi")

# Global transport controls
def render_global_transport():
    st.markdown('<div class="global-transport">', unsafe_allow_html=True)
    if st.button("⏵︎", key='play'):
        st.session_state.playback['is_playing'] = True
    if st.button("⏸︎", key='pause'):
        st.session_state.playback['is_playing'] = False
    if st.button("⏹︎", key='stop'):
        st.session_state.playback['is_playing'] = False
        st.session_state.playback['current_position'] = 0
    st.markdown('</div>', unsafe_allow_html=True)

render_global_transport()
