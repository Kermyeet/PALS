import numpy as np
from scipy.io import wavfile
import pyaudio as pa
import struct
import matplotlib.pyplot as plt

song = 'Moby Dick.wav'

# Read the .wav file
fs, data = wavfile.read(song)

FORMAT = pa.paInt16