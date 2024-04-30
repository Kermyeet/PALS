import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft
from pydub import AudioSegment
import sounddevice as sd

# constants
CHUNK = 1024           # samples per frame

# load the song file
song = AudioSegment.from_file("Moby Dick.wav")

RATE = song.frame_rate      

# convert song to numpy array
song_array = np.array(song.get_array_of_samples())

# check if stereo
if song.channels == 2:
    song_array = song_array.reshape((-1, 2))
    
fft_array = np.empty((len(song_array) // CHUNK, CHUNK), dtype=np.complex64)
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0])
        yf2 = fft(data[:, 1])
        fft_array[i] = (yf1 + yf2) / 2
    else:
        fft_array[i] = fft(data)



# Reduce number of points in FFT
NFFT = 1024  # adjust this value as needed

xf = np.linspace(0, RATE, NFFT)     # frequencies (spectrum)

fig, ax = plt.subplots(figsize=(15, 7))

line_fft, = ax.semilogx(xf, np.random.rand(CHUNK), '-', lw=2)

ax.set_xlim(20, RATE / 2)

xticks = np.logspace(np.log10(20), np.log10(RATE / 2), num=9)
ax.set_xticks(xticks)

ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_ylim([0, 5])

fft_array = np.empty((len(song_array) // CHUNK, CHUNK), dtype=np.complex64)
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0])
        yf2 = fft(data[:, 1])
        fft_array[i] = (yf1 + yf2) / 2
    else:
        fft_array[i] = fft(data)


def animate(i):
    line_fft.set_ydata(np.abs(fft_array[i][0:1024])  / (1024 * CHUNK))

sd.play(song_array / np.max(np.abs(song_array)), RATE)

interval = (CHUNK / RATE) * 1  # in milliseconds

# create animation
anim = animation.FuncAnimation(fig, animate, frames=len(song_array)//CHUNK, interval=interval)

plt.show()

