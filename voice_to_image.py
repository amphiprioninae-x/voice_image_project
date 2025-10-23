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
# Rainbow colors from outer to inner
RAINBOW_COLORS = [
    (255, 0, 0),     # Red
    (255, 127, 0),   # Orange
    (255, 255, 0),   # Yellow
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (75, 0, 130),    # Indigo
    (148, 0, 211)    # Violet
]

# Audio settings
sample_rate = 44100  # Hz
block_size = 1024    # Standard buffer size
channels = 1         # Mono
device = 21         # Using Windows WASAPI microphone
VOLUME_MULTIPLIER = 1000.0  # Increased sensitivity significantly
NOISE_THRESHOLD = 0.001    # Lowered noise threshold

# Debug flag
DEBUG = True        # Set to True to see volume levels

# Print available audio devices at startup
print("\nAvailable audio devices:")
print(sd.query_devices())
print(f"\nUsing audio device {device}: {sd.query_devices(device)['name']}")

# Animation settings
animation_time = 0.0
base_radius = 20
ring_spacing = 4  # Space between rings

# Initial position and animation state
dot_x, dot_y = WIDTH // 2, HEIGHT // 2
ring_radiuses = [base_radius + i * ring_spacing for i in range(7)]
current_volume = 0.0
target_volume = 0.0
VOLUME_SMOOTH_FACTOR = 0.15  # Controls how smoothly the volume changes

# Font for displaying volume level
font = pygame.font.Font(None, 36)

def audio_callback(indata, frames, time, status):
    """This is called for each audio block."""
    global dot_x, dot_y, ring_radiuses, animation_time, current_volume, target_volume
    
    if status:
        print(f"Status: {status}")
    
    # Get absolute values of audio data and calculate volume
    audio_data = np.abs(indata.flatten())
    volume = np.sqrt(np.mean(audio_data**2))
    volume_normalized = max(0, min(1.0, volume * VOLUME_MULTIPLIER))
    
    # Update animation time
    animation_time += frames / sample_rate
    
    # Update position and volume based on audio input
    if volume > NOISE_THRESHOLD:
        # Update volume target
        target_volume = volume_normalized
        
        # Update position
        angle = animation_time * 3
        movement_radius = volume_normalized * (min(WIDTH, HEIGHT) // 3) * 1.5
        target_x = WIDTH // 2 + math.cos(angle) * movement_radius
        target_y = HEIGHT // 2 + math.sin(angle) * movement_radius
        
        # Smooth movement
        dot_x = dot_x * 0.7 + target_x * 0.3
        dot_y = dot_y * 0.7 + target_y * 0.3
    else:
        # Return to center when no sound
        dot_x = WIDTH // 2
        dot_y = HEIGHT // 2
        target_volume = 0.0
    
    # Smooth volume transition
    current_volume += (target_volume - current_volume) * VOLUME_SMOOTH_FACTOR
    
    # Debug output
    if DEBUG:
        print(f"\rVolume: {volume_normalized:.2%}, Current: {current_volume:.2f}", end="")

def main():
    global device

    try:
        # Set default input device
        device = sd.default.device[0]
    except Exception as e:
        print(f"Error during setup: {e}")
        return

    try:
        # Start audio stream
        stream = sd.InputStream(
            callback=audio_callback,
            device=device,
            channels=channels,
            samplerate=sample_rate,
            blocksize=block_size,
            dtype=np.float32
        )
        
        with stream:
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
                
                # Draw rainbow rings from outer to inner
                for i, base_radius in enumerate(ring_radiuses):
                    # Calculate radius based on current volume
                    volume_scale = 1.0 + (current_volume * 2.0)  # Scale from 1.0 to 3.0
                    scaled_radius = int(base_radius * volume_scale)
                    # Only draw if radius is positive
                    if scaled_radius > 0:
                        pygame.draw.circle(screen, RAINBOW_COLORS[i], (int(dot_x), int(dot_y)), scaled_radius, 2)
                
                # Draw volume indicator
                volume_percentage = int(current_volume * 100)  # Convert to percentage
                volume_text = font.render(f"Volume: {volume_percentage}%", True, WHITE)
                screen.blit(volume_text, (10, 10))  # Top-left corner
                
                # Display instructions
                instructions = font.render("Speak to see the rings move. Press ESC to exit.", True, WHITE)
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
