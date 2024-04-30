import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft

# constants
CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# variable for plotting
xf = np.linspace(0, RATE, CHUNK)     # frequencies (spectrum)

# create a figure with a single subplot
fig, ax = plt.subplots(figsize=(15, 7))

# create semilogx line for spectrum
line_fft, = ax.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

# format spectrum axes
ax.set_xlim(20, RATE / 2)

def animate(i):
    # get audio data
    data = stream.read(CHUNK)  
    data_np = np.frombuffer(data, dtype='h')

    # compute FFT and update line
    yf = fft(data_np)
    line_fft.set_ydata(np.abs(yf[0:CHUNK])  / (512 * CHUNK))

# create animation
anim = animation.FuncAnimation(fig, animate, interval=1)

plt.show()