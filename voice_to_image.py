import pygame
import numpy as np
import sounddevice as sd
import math
import time

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Voice Visualizer")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Audio settings
sample_rate = 44100  # Hz
block_size = 1024    # Buffer size
channels = 1         # Mono

# Initial dot position (center of screen)
dot_x, dot_y = WIDTH // 2, HEIGHT // 2
dot_radius = 20

# Font for displaying volume level
font = pygame.font.Font(None, 36)

def audio_callback(indata, frames, time, status):
    """This is called for each audio block."""
    global dot_x, dot_y, dot_radius
    
    if status:
        print(f"Status: {status}")
        
    # Calculate volume (root mean square of audio samples)
    volume = np.sqrt(np.mean(indata**2))
    
    # Scale volume (adjust these values as needed)
    volume_normalized = min(1.0, volume * 5)  # Scale factor may need adjustment
    
    # Update dot position based on volume
    # Make the dot move in a circular pattern with radius proportional to volume
    angle = time.currentTime * 2  # Rotation speed
    radius = volume_normalized * (min(WIDTH, HEIGHT) // 3)
    
    # Calculate new position
    target_x = WIDTH // 2 + math.cos(angle) * radius
    target_y = HEIGHT // 2 + math.sin(angle) * radius
    
    # Smooth movement (lerp)
    dot_x = dot_x * 0.9 + target_x * 0.1
    dot_y = dot_y * 0.9 + target_y * 0.1
    
    # Update dot size based on volume
    dot_radius = 10 + int(volume_normalized * 40)

def main():
    # Start audio stream
    try:
        with sd.InputStream(callback=audio_callback, 
                           channels=channels,
                           samplerate=sample_rate,
                           blocksize=block_size):
            
            print("Voice visualizer started. Speak into the microphone.")
            
            running = True
            clock = pygame.time.Clock()
            
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                
                # Clear screen
                screen.fill(BLACK)
                
                # Draw dot
                color = (
                    min(255, int(dot_radius * 5)),
                    min(255, int(255 - dot_radius * 2)),
                    min(255, int(128 + dot_radius * 2))
                )
                pygame.draw.circle(screen, color, (int(dot_x), int(dot_y)), dot_radius)
                
                # Draw volume indicator
                volume_text = font.render(f"Volume: {int((dot_radius - 10) / 40 * 100)}%", True, WHITE)
                screen.blit(volume_text, (10, 10))
                
                # Display instructions
                instructions = font.render("Speak to see the dot move. Press ESC to exit.", True, WHITE)
                screen.blit(instructions, (10, HEIGHT - 40))
                
                # Update display
                pygame.display.flip()
                
                # Limit to 60 FPS
                clock.tick(60)
                
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
