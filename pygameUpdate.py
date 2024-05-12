import pygame
import numpy as np
from scipy.fftpack import fft
from pydub import AudioSegment
import sounddevice as sd
import sys

CHUNK = 2048 

song = AudioSegment.from_file("Moby Dick.wav")

song = song.set_frame_rate(44100)

RATE = song.frame_rate

song_array = np.array(song.get_array_of_samples())

if song.channels == 2:
    song_array = song_array.reshape((-1, 2))

pygame.init()
WIDTH = 1500;
HEIGHT = 1000;
screen = pygame.display.set_mode((WIDTH, HEIGHT))
MAX_HEIGHT = 0.8 * HEIGHT 

def draw_spectrum(fft_array):
    screen.fill((0, 0, 0)) 
    heights = []
    max_fft = np.max(np.abs(fft_array))
    
    max_index = int(20000 * (CHUNK / RATE))   

    for i in range(1, min(max_index, CHUNK)): 
        height = np.abs(fft_array[i]) / max_fft * MAX_HEIGHT  # scale to MAX_HEIGHT
        x = int(np.log(i) / np.log(max_index) * WIDTH) 
        pygame.draw.line(screen, (0, 255, 0), (x, HEIGHT), (x, HEIGHT - height))
        heights.append((x, HEIGHT - height))
    pygame.draw.lines(screen, (255, 0, 0), False, heights, 1) 

    font = pygame.font.Font(None, 24) 

    label_left = font.render("20 Hz", 1, (255, 255, 255))
    screen.blit(label_left, (0, HEIGHT - 20)) 

    label_right = font.render(str(RATE / 2) + " Hz", 1, (255, 255, 255))
    screen.blit(label_right, (WIDTH - 100, HEIGHT - 20)) 

    for i in range(1, 9):  # draw 8 labels
        freq = int(20 * np.exp(i / 8 * np.log((RATE / 2) / 20)))  
        label = font.render(str(freq) + " Hz", 1, (255, 255, 255)) 
        screen.blit(label, (i / 8 * WIDTH, HEIGHT - 20)) 

# precompute FFT for each chunk
fft_arrays = []
window = np.hanning(CHUNK)  # create a window
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0] * window)  
        yf2 = fft(data[:, 1] * window)  
        fft_array = (yf1 + yf2) / 2
    else:
        fft_array = fft(data * window)  
    fft_arrays.append(fft_array)

# play the song and update the spectrum analyzer
sd.play(song_array / np.max(np.abs(song_array)), RATE)
pygame.time.set_timer(pygame.USEREVENT, int(1000 * CHUNK / RATE))  
fft_index = 0

slider_rect = pygame.Rect(50, 50, 200, 50)  
slider_pos = 1  
dragging = False  

while True:
    for event in pygame.event.get():
        if event.type == pygame.USEREVENT:
            font = pygame.font.Font(None, 24) 

            draw_spectrum(fft_arrays[fft_index])
            pygame.draw.rect(screen, (200, 200, 200), slider_rect)  
            pygame.draw.circle(screen, (255, 0, 0), (slider_rect.left + int(slider_pos * slider_rect.width), slider_rect.centery), 20) 
            chunk_label = font.render("Zoom: " + str(CHUNK), 1, (255, 255, 255))
            screen.blit(chunk_label, (slider_rect.centerx, slider_rect.top - 20))  
            pygame.display.flip()  # update the screen
            fft_index += 1
        elif event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if slider_rect.collidepoint(event.pos):
                dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
        elif event.type == pygame.MOUSEMOTION and dragging:
            slider_pos = (event.pos[0] - slider_rect.left) / slider_rect.width
            slider_pos = max(0, min(1, slider_pos))  

            CHUNK = max(7, int(slider_pos * 1024))
            

