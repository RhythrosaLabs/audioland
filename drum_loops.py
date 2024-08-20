# drum_loop_generator.py
# DrumLoopGenerator class for generating drum loops

import random
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

class DrumLoopGenerator:
    def __init__(self, tempo=120, beat_length=16):
        self.tempo = tempo
        self.beat_length = beat_length

    def _generate_kick(self):
        freq = random.uniform(40, 80)
        duration = 100
        return Sine(freq).to_audio_segment(duration=duration)

    def _generate_snare(self):
        freq = random.uniform(600, 2000)
        duration = 100
        snare = Sine(freq).to_audio_segment(duration=duration)
        return snare + WhiteNoise().to_audio_segment(duration=duration)

    def _generate_hihat(self):
        return WhiteNoise().to_audio_segment(duration=25)

    def _generate_tom(self):
        freq_range = random.choice([(100, 150), (150, 250), (250, 350)])
        freq = random.uniform(*freq_range)
        duration = 100
        return Sine(freq).to_audio_segment(duration=duration)

    def _generate_silence(self):
        return AudioSegment.silent(duration=100)

    def generate_loop(self, filename):
        beat_duration = 60000 // self.tempo
        loop = AudioSegment.silent(duration=beat_duration * self.beat_length)

        for i in range(self.beat_length):
            if i % 4 == 0:
                loop = loop.overlay(self._generate_kick(), position=i * beat_duration)
            elif i % 4 == 2:
                loop = loop.overlay(self._generate_snare(), position=i * beat_duration)
            if random.random() > 0.7:
                loop = loop.overlay(self._generate_hihat(), position=i * beat_duration)
            if random.random() > 0.85:
                loop = loop.overlay(self._generate_tom(), position=i * beat_duration)

        loop.export(filename, format='wav')

# Example usage
if __name__ == "__main__":
    generator = DrumLoopGenerator()
    generator.generate_loop("drum_loop.wav")
