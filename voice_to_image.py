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
block_size = 2048    # Increased buffer size to prevent overflow
channels = 1         # Mono
device = 2          # Using Realtek Audio microphone (you can change this number based on available devices)

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
    
    # Print audio data stats occasionally
    if np.random.random() < 0.01:  # Print roughly every 100th frame
        print(f"Audio input stats - Max: {np.max(indata):.3f}, Mean: {np.mean(indata):.3f}, Min: {np.min(indata):.3f}")
    # Calculate volume (root mean square of audio samples)
    volume = np.sqrt(np.mean(indata**2))
    
    # Scale volume with higher sensitivity (increased from 5 to 20)
    volume_normalized = min(1.0, volume * 20)  # Increased scale factor for more sensitivity
    
    # Update dot position based on volume
    # Make the dot move in a circular pattern with radius proportional to volume
    angle = time.currentTime * 3  # Increased rotation speed
    radius = volume_normalized * (min(WIDTH, HEIGHT) // 3)
    
    # Calculate new position with larger movement range
    radius = radius * 1.5  # Increase the movement radius by 50%
    target_x = WIDTH // 2 + math.cos(angle) * radius
    target_y = HEIGHT // 2 + math.sin(angle) * radius
    
    # Faster movement response (adjusted lerp values)
    dot_x = dot_x * 0.7 + target_x * 0.3  # More responsive horizontal movement
    dot_y = dot_y * 0.7 + target_y * 0.3  # More responsive vertical movement
    
    # Update dot size based on volume with more dramatic changes
    dot_radius = 10 + int(volume_normalized * 80)  # Doubled the size range

def main():
    # Print available audio devices
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    
    # Get default input device
    device_info = sd.query_devices(kind='input')
    print("\nDefault input device:")
    print(device_info)
    
    # Start audio stream
    try:
        print("\nAttempting to open audio stream...")
        with sd.InputStream(callback=audio_callback, 
                           device=device,
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
