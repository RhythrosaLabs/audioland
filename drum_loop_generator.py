import random
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
import io
import numpy as np

class DrumLoopGenerator:
    def __init__(self, tempo=120, beat_length=16):
        self.tempo = tempo
        self.beat_length = beat_length
        self.sample_rate = 44100

    def _create_audio_segment(self, samples):
        return AudioSegment(
            samples.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        )

    def _generate_kick(self):
        freq = random.uniform(50, 70)
        duration = 150
        t = np.linspace(0, duration / 1000, int(self.sample_rate * duration / 1000))
        sine = np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-t * 30)
        kick = sine * envelope
        return self._create_audio_segment(np.int16(kick * 32767))

    def _generate_snare(self):
        freq = random.uniform(150, 250)
        duration = 100
        t = np.linspace(0, duration / 1000, int(self.sample_rate * duration / 1000))
        sine = np.sin(2 * np.pi * freq * t)
        noise = np.random.rand(len(t)) * 2 - 1
        envelope = np.exp(-t * 40)
        snare = (sine + noise) * envelope
        return self._create_audio_segment(np.int16(snare * 16383))

    def _generate_hihat(self):
        duration = 50
        t = np.linspace(0, duration / 1000, int(self.sample_rate * duration / 1000))
        noise = np.random.rand(len(t)) * 2 - 1
        envelope = np.exp(-t * 200)
        hihat = noise * envelope
        return self._create_audio_segment(np.int16(hihat * 8191))

    def _generate_tom(self):
        freq_range = random.choice([(70, 100), (100, 130), (130, 160)])
        freq = random.uniform(*freq_range)
        duration = 150
        t = np.linspace(0, duration / 1000, int(self.sample_rate * duration / 1000))
        sine = np.sin(2 * np.pi * freq * t)
        envelope = np.exp(-t * 25)
        tom = sine * envelope
        return self._create_audio_segment(np.int16(tom * 16383))

    def generate_loop(self):
        beat_duration = 60000 // self.tempo
        loop = AudioSegment.silent(duration=beat_duration * self.beat_length, frame_rate=self.sample_rate)
        
        for i in range(self.beat_length):
            if i % 4 == 0:
                loop = loop.overlay(self._generate_kick(), position=i * beat_duration)
            if i % 4 == 2:
                loop = loop.overlay(self._generate_snare(), position=i * beat_duration)
            if random.random() > 0.5:
                loop = loop.overlay(self._generate_hihat(), position=i * beat_duration)
            if random.random() > 0.8:
                loop = loop.overlay(self._generate_tom(), position=i * beat_duration)
        
        buffer = io.BytesIO()
        loop = loop.apply_gain(-3)  # Reduce overall volume
        loop.export(buffer, format='wav')
        buffer.seek(0)
        return buffer

if __name__ == "__main__":
    generator = DrumLoopGenerator()
    generator.generate_loop()
