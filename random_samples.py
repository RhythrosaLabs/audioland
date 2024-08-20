import numpy as np
from scipy.io.wavfile import write
import scipy.signal
import zipfile
import os
import io

# Constants
SAMPLE_RATE = 44100  # Sampling rate in Hz
DURATION = 2  # Duration of each sound effect in seconds
NUM_FILES = 25  # Number of sound effects to generate
SEQUENCE_LENGTH = 30  # Length of the musical sequence in seconds

# We'll use BytesIO instead of temporary files
def apply_reverb(signal, decay=0.5):
    """Apply a reverb effect with decay."""
    reverb = np.concatenate([signal, np.zeros(int(SAMPLE_RATE * 0.5))])
    reverb[int(SAMPLE_RATE * 0.1):] *= decay
    return reverb[:len(signal)]

def apply_delay(signal, delay_ms=100):
    """Apply a delay effect."""
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    delayed_signal = np.concatenate([np.zeros(delay_samples), signal])
    return delayed_signal[:len(signal)]

def apply_lpf(signal, cutoff=3000):
    """Apply a low-pass filter effect."""
    sos = scipy.signal.butter(10, cutoff, 'low', fs=SAMPLE_RATE, output='sos')
    return scipy.signal.sosfilt(sos, signal)

def apply_hpf(signal, cutoff=100):
    """Apply a high-pass filter effect."""
    sos = scipy.signal.butter(10, cutoff, 'high', fs=SAMPLE_RATE, output='sos')
    return scipy.signal.sosfilt(sos, signal)

def apply_phaser(signal):
    """Apply a simple phaser effect."""
    return scipy.signal.lfilter([1, -0.9], [1, -0.9], signal)

def apply_chorus(signal, depth=0.5, rate=0.5):
    """Apply a chorus effect."""
    t = np.arange(len(signal)) / SAMPLE_RATE
    mod = depth * np.sin(2 * np.pi * rate * t)
    chorus_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        if i - int(mod[i]) >= 0:
            chorus_signal[i] = signal[i] + signal[i - int(mod[i])]
    return chorus_signal

def apply_distortion(signal, gain=1.5):
    """Apply a distortion effect."""
    distorted = np.tanh(gain * signal)
    return distorted

def apply_flanger(signal, depth=0.5, rate=0.25):
    """Apply a flanger effect."""
    delay_samples = int(SAMPLE_RATE * 0.0025)
    t = np.arange(len(signal)) / SAMPLE_RATE
    lfo = depth * np.sin(2 * np.pi * rate * t)
    flanged_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        delay = int(delay_samples * (1 + lfo[i]))
        if i - delay >= 0:
            flanged_signal[i] = signal[i] + signal[i - delay]
    return flanged_signal

def apply_echo(signal, delay_ms=300, decay=0.5):
    """Apply a longer echo effect."""
    delay_samples = int(SAMPLE_RATE * delay_ms / 1000)
    echo_signal = np.zeros_like(signal)
    for i in range(len(signal)):
        if i - delay_samples >= 0:
            echo_signal[i] = signal[i] + decay * signal[i - delay_samples]
        else:
            echo_signal[i] = signal[i]
    return echo_signal

def generate_arpeggio(t, base_freq):
    """Generate an arpeggio effect."""
    freqs = [base_freq, base_freq * 1.5, base_freq * 2]
    arpeggio = np.zeros_like(t)
    for i, freq in enumerate(freqs):
        arpeggio += 0.2 * np.sin(2 * np.pi * freq * t * (i + 1))
    return arpeggio

def generate_random_sound():
    """Generate a complex random sound effect."""
    sound_type = np.random.choice(['arpeggio', 'chord', 'bloop', 'bleep'])
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    audio = np.zeros(t.shape[0])

    if sound_type == 'arpeggio':
        base_freq = np.random.uniform(100, 1000)
        audio += generate_arpeggio(t, base_freq)
    elif sound_type == 'chord':
        freq1 = np.random.uniform(100, 500)
        freq2 = freq1 * 1.5
        freq3 = freq1 * 2
        audio += 0.4 * np.sin(2 * np.pi * freq1 * t)
        audio += 0.3 * np.sin(2 * np.pi * freq2 * t)
        audio += 0.2 * np.sin(2 * np.pi * freq3 * t)
    elif sound_type == 'bloop':
        freq = np.random.uniform(200, 800)
        audio += 0.5 * np.sin(2 * np.pi * freq * t)
        audio[int(SAMPLE_RATE * DURATION / 4):int(SAMPLE_RATE * DURATION / 2)] *= 2
    elif sound_type == 'bleep':
        freq = np.random.uniform(500, 1500)
        audio += 0.5 * np.sin(2 * np.pi * freq * t)
        audio[int(SAMPLE_RATE * DURATION / 2):] = 0

    # Apply effects
    audio = apply_reverb(audio)
    audio = apply_delay(audio)
    audio = apply_lpf(audio)
    audio = apply_hpf(audio)
    audio = apply_phaser(audio)
    audio = apply_chorus(audio)
    audio = apply_distortion(audio)
    audio = apply_flanger(audio)
    audio = apply_echo(audio)
    
    # Normalize
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    
    return audio

def create_random_samples():
    """Create random audio samples and return as a list of BytesIO objects."""
    samples = []
    for _ in range(NUM_FILES):
        sound_data = generate_random_sound()
        buffer = io.BytesIO()
        write(buffer, SAMPLE_RATE, sound_data)
        buffer.seek(0)
        samples.append(buffer)
    return samples

def create_musical_sequence(samples):
    """Create a musical sequence from the generated sound samples."""
    sequence = np.zeros(int(SAMPLE_RATE * SEQUENCE_LENGTH))
    
    for i in range(SEQUENCE_LENGTH):
        sound_idx = np.random.randint(len(samples))
        samples[sound_idx].seek(0)
        _, sound_data = scipy.io.wavfile.read(samples[sound_idx])
        
        start_idx = int(i * SAMPLE_RATE)
        end_idx = start_idx + len(sound_data)
        if end_idx < len(sequence):
            sequence[start_idx:end_idx] += sound_data
        
    sequence = np.int16(sequence / np.max(np.abs(sequence)) * 32767)
    buffer = io.BytesIO()
    write(buffer, SAMPLE_RATE, sequence)
    buffer.seek(0)
    return buffer

def generate_random_samples_and_sequence():
    """Generate random samples and a musical sequence, return as BytesIO objects."""
    samples = create_random_samples()
    sequence = create_musical_sequence(samples)
    return samples, sequence
