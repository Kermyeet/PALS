import pygame
import numpy as np
from scipy.fftpack import fft
from pydub import AudioSegment
import sys

# Change CHUNK to a higher value for balanced out data, change to lower for more accurate but sporatic data
CHUNK = 512 * 2

song = AudioSegment.from_file("Moby Dick.wav")

RATE = song.frame_rate

song_array = np.array(song.get_array_of_samples())

if song.channels == 2:
    song_array = song_array.reshape((-1, 2))

pygame.init()
WIDTH = 1500
HEIGHT = 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
MAX_HEIGHT = 0.8 * HEIGHT 

def draw_spectrum(fft_array):
    screen.fill((0, 0, 0)) 
    heights = []
    max_fft = np.max(np.abs(fft_array))
    
    max_index = int(22050 * (CHUNK / RATE))   

    for i in range(1, min(max_index, CHUNK)): 
        height = np.abs(fft_array[i]) / max_fft * MAX_HEIGHT 
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
window = np.kaiser(CHUNK, 14)  # create a window
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0] * window)  
        yf2 = fft(data[:, 1] * window)  
        fft_array = (yf1 + yf2) / 2
    else:
        fft_array = fft(data * window)  
    fft_arrays.append(fft_array)

fft_hanning = []
window = np.hanning(CHUNK)  # create a window
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0] * window)  
        yf2 = fft(data[:, 1] * window)  
        fft_array = (yf1 + yf2) / 2
    else:
        fft_array = fft(data * window)  
    fft_hanning.append(fft_array)

fft_hamming = []
window = np.hamming(CHUNK)  # create a window
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0] * window)  
        yf2 = fft(data[:, 1] * window)  
        fft_array = (yf1 + yf2) / 2
    else:
        fft_array = fft(data * window)  
    fft_hamming.append(fft_array)

fft_blackman = []
window = np.blackman(CHUNK)  # create a window
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0] * window)  
        yf2 = fft(data[:, 1] * window)  
        fft_array = (yf1 + yf2) / 2
    else:
        fft_array = fft(data * window)  
    fft_blackman.append(fft_array)

fft_bartlett = []
window = np.bartlett(CHUNK)  # create a window
for i in range(len(song_array) // CHUNK):
    data = song_array[i*CHUNK:(i+1)*CHUNK]
    if song.channels == 2:
        yf1 = fft(data[:, 0] * window)  
        yf2 = fft(data[:, 1] * window)  
        fft_array = (yf1 + yf2) / 2
    else:
        fft_array = fft(data * window)  
    fft_bartlett.append(fft_array)

def use_kaiser():
    global current_fft
    global current_window
    current_fft = fft_arrays
    current_window = 'Kaiser'

def use_hanning():
    global current_fft
    global current_window
    current_fft = fft_hanning
    current_window = 'Hanning'

def use_hamming():
    global current_fft
    global current_window
    current_fft = fft_hamming
    current_window = 'Hamming'

def use_bartlett():
    global current_fft
    global current_window
    current_fft = fft_bartlett
    current_window = 'Bartlett'

def use_blackman():
    global current_fft
    global current_window
    current_fft = fft_blackman
    current_window = 'Blackman'

def text_objects(text, font):
    textSurface = font.render(text, True, (0,0,0))
    return textSurface, textSurface.get_rect()

def draw_button(screen, message, x, y, w, h, ic, ac, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(screen, ac,(x,y,w,h))
        if click[0] == 1 and action != None:
            action() 
    else:
        pygame.draw.rect(screen, ic,(x,y,w,h))

    smallText = pygame.font.SysFont(None,20)
    textSurf, textRect = text_objects(message, smallText)
    textRect.center = ( (x+(w/2)), (y+(h/2)) )
    screen.blit(textSurf, textRect)

def draw_text_box(screen, message, x, y, w, h, color):
    pygame.draw.rect(screen, color, (x, y, w, h))
    smallText = pygame.font.SysFont(None, 20)
    textSurf, textRect = text_objects(message, smallText)
    textRect.center = ((x + (w / 2)), (y + (h / 2)))
    screen.blit(textSurf, textRect)

current_window = 'Kaiser'
current_fft = fft_arrays
volume = 1.0  # Volume level from 0 to 1

# Initialize song playback and set up the timer
song = pygame.mixer.Sound('Moby Dick.wav')
chunk_duration_ms = int((CHUNK / RATE) * 1000)
pygame.time.set_timer(pygame.USEREVENT, chunk_duration_ms)  
song.play()

fft_index = 0
slider_rect = pygame.Rect(50, 50, 200, 50)  
slider_pos = 1  
dragging_zoom = False  

volume_slider_rect = pygame.Rect(50, 120, 200, 50)  
volume_pos = 1  
dragging_volume = False  

while True:
    for event in pygame.event.get():
        if event.type == pygame.USEREVENT:
            font = pygame.font.Font(None, 24) 

            draw_spectrum(current_fft[fft_index])
            pygame.draw.rect(screen, (200, 200, 200), slider_rect)  
            pygame.draw.circle(screen, (255, 0, 0), (slider_rect.left + int(slider_pos * slider_rect.width), slider_rect.centery), 20) 
            chunk_label = font.render("Zoom: " + str(CHUNK), 1, (255, 255, 255))
            screen.blit(chunk_label, (slider_rect.centerx, slider_rect.top - 20))  

            pygame.draw.rect(screen, (200, 200, 200), volume_slider_rect)  
            pygame.draw.circle(screen, (255, 0, 0), (volume_slider_rect.left + int(volume_pos * volume_slider_rect.width), volume_slider_rect.centery), 20) 
            volume_label = font.render("Volume: " + str(round(volume * 100)) + "%", 1, (255, 255, 255))
            screen.blit(volume_label, (volume_slider_rect.centerx, volume_slider_rect.top - 20))  

            pygame.display.flip()  # update the screen
            fft_index = (fft_index + 1) % len(current_fft)
        elif event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if slider_rect.collidepoint(event.pos):
                dragging_zoom = True
            if volume_slider_rect.collidepoint(event.pos):
                dragging_volume = True
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging_zoom = False
            dragging_volume = False
        elif event.type == pygame.MOUSEMOTION:
            if dragging_zoom:
                slider_pos = (event.pos[0] - slider_rect.left) / slider_rect.width
                slider_pos = max(0, min(1, slider_pos))  
                CHUNK = max(7, int(slider_pos * 1024))
            if dragging_volume:
                volume_pos = (event.pos[0] - volume_slider_rect.left) / volume_slider_rect.width
                volume_pos = max(0, min(1, volume_pos))
                volume = volume_pos
                song.set_volume(volume)
                
    draw_button(screen, 'Kaiser', 50 + 300, 50, 100, 50, (0,200,0), (0,255,0), use_kaiser)
    draw_button(screen, 'Hanning', 200 + 300, 50, 100, 50, (0,200,0), (0,255,0), use_hanning)    
    draw_button(screen, 'Hamming', 350 + 300, 50, 100, 50, (0,200,0), (0,255,0), use_hamming)  
    draw_button(screen, 'Bartlett', 500 + 300, 50, 100, 50, (0,200,0), (0,255,0), use_bartlett)
    draw_button(screen, 'Blackman', 650 + 300, 50, 100, 50, (0,200,0), (0,255,0), use_blackman)
    draw_text_box(screen, 'Current window: ' + current_window, 1200, 50, 200, 50, (200, 200, 200))

    pygame.display.flip()
