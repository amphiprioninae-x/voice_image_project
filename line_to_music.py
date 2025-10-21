import pygame
import numpy as np
import math
from datetime import datetime
import wave
import struct
import os

# Initialize pygame
pygame.init()

# Create output directories if they don't exist
PICTURE_DIR = "creation/picture"
MUSIC_DIR = "creation/music"
os.makedirs(PICTURE_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)

# Screen dimensions
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Line to Music")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BLUE = (120, 180, 255)
LIGHT_GREEN = (144, 238, 144)
LIGHT_RED = (255, 128, 128)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)

# Drawing area
CANVAS_TOP = 100
CANVAS_HEIGHT = HEIGHT - CANVAS_TOP
drawing_surface = pygame.Surface((WIDTH, CANVAS_HEIGHT))
drawing_surface.fill(WHITE)

# Font
font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 24)

# Drawing variables
drawing = False
points = []  # Store all points with timestamps
current_line = []  # Current line being drawn
all_lines = []  # All completed lines
line_start_time = 0

# Audio settings
SAMPLE_RATE = 44100
DURATION = 8  # seconds

# Humanization controls
HUMANIZE_ENABLED = True
HUMANIZE_INTENSITY = 0.15  # 0.0-0.4 recommended (duration/volume jitter ±15%)
SWING_FACTOR = 0.10        # 10% swing: lengthen odd beats, shorten even beats

# Musical scales (C major scale frequencies in Hz)
C_MAJOR_SCALE = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5
PENTATONIC_SCALE = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]  # C major pentatonic

# Style presets
STYLES = {
    "Pentatonic (Major)": {
        "scale": PENTATONIC_SCALE,
        "pad_gain": 1.0,
        "perc_gain": 1.0,
    },
    "Pentatonic (Minor)": {
        # A minor pentatonic around A3-A4 region
        "scale": [220.00, 261.63, 293.66, 329.63, 392.00, 440.00],
        "pad_gain": 0.9,
        "perc_gain": 1.1,
    },
    "Blues (C)": {
        # C blues: C Eb F Gb G Bb C
        "scale": [261.63, 311.13, 349.23, 369.99, 392.00, 466.16, 523.25],
        "pad_gain": 0.95,
        "perc_gain": 1.2,
    },
    "Major (C)": {
        "scale": C_MAJOR_SCALE,
        "pad_gain": 1.1,
        "perc_gain": 0.9,
    },
    "Ambient": {
        # Use major pentatonic but softer percussive feel and stronger pad
        "scale": PENTATONIC_SCALE,
        "pad_gain": 1.4,
        "perc_gain": 0.6,
    },
}

STYLE_NAMES = list(STYLES.keys())
style_index = 0
CURRENT_STYLE = STYLE_NAMES[style_index]
CURRENT_STYLE_CONFIG = STYLES[CURRENT_STYLE]
CURRENT_SCALE = CURRENT_STYLE_CONFIG["scale"]

class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover = False
    
    def draw(self, surface):
        color = tuple(min(c + 30, 255) for c in self.color) if self.hover else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, BLACK, self.rect, 2, border_radius=5)
        text_surf = small_font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# Create buttons
clear_button = Button(20, 20, 150, 50, "Clear the board", LIGHT_RED)
save_drawing_button = Button(190, 20, 180, 50, "Download the picture", LIGHT_BLUE)
generate_music_button = Button(390, 20, 170, 50, "Create the music", LIGHT_GREEN)
save_music_button = Button(580, 20, 180, 50, "Download the sound", LIGHT_GREEN)
style_button = Button(780, 20, 200, 50, f"Style: {CURRENT_STYLE}", LIGHT_BLUE)

buttons = [clear_button, save_drawing_button, generate_music_button, save_music_button, style_button]

# Music generation status
music_generated = False
audio_data = None

def save_drawing_as_image():
    """Save the current drawing as PNG"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drawing_{timestamp}.png"
    filepath = os.path.join(PICTURE_DIR, filename)
    pygame.image.save(drawing_surface, filepath)
    print(f"绘画已保存为: {filepath}")
    return filename

def calculate_line_properties(points):
    """Calculate properties of the drawn line for music generation"""
    if len(points) < 2:
        return None
    
    # Calculate total length
    total_length = 0
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        total_length += math.sqrt(dx*dx + dy*dy)
    
    # Calculate drawing time
    drawing_time = points[-1][2] - points[0][2]
    
    # Calculate average speed (pixels per second)
    speed = total_length / drawing_time if drawing_time > 0 else 0
    
    # Calculate curvature (how much the line changes direction)
    angles = []
    for i in range(1, len(points) - 1):
        dx1 = points[i][0] - points[i-1][0]
        dy1 = points[i][1] - points[i-1][1]
        dx2 = points[i+1][0] - points[i][0]
        dy2 = points[i+1][1] - points[i][1]
        
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        angle_diff = abs(angle2 - angle1)
        angles.append(angle_diff)
    
    avg_curvature = sum(angles) / len(angles) if angles else 0
    
    # Calculate height variation (for pitch)
    y_values = [p[1] for p in points]
    y_min, y_max = min(y_values), max(y_values)
    height_range = y_max - y_min
    
    return {
        'length': total_length,
        'drawing_time': drawing_time,
        'speed': speed,
        'curvature': avg_curvature,
        'y_min': y_min,
        'y_max': y_max,
        'height_range': height_range,
        'points': points
    }

def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE, amplitude=0.3, harmonics=True,
                  vibrato_depth=0.0, vibrato_rate=5.0, env_attack_ms=10, env_release_ms=10):
    """Generate a richer tone with harmonics and optional vibrato and variable envelope"""
    n = int(sample_rate * duration)
    if n <= 0:
        return np.array([], dtype=float)
    t = np.linspace(0, duration, n, False)
    # Envelope
    attack = max(1, int(sample_rate * (env_attack_ms / 1000.0)))
    release = max(1, int(sample_rate * (env_release_ms / 1000.0)))
    envelope = np.ones_like(t)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    # Vibrato (frequency modulation)
    if vibrato_depth > 0.0 and vibrato_rate > 0.0:
        phase = 2 * np.pi * (frequency * t + frequency * vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
    else:
        phase = 2 * np.pi * frequency * t
    
    base = np.sin(phase)
    if harmonics:
        wave = (amplitude * base +
                amplitude * 0.3 * np.sin(2 * phase) +
                amplitude * 0.2 * np.sin(3 * phase) +
                amplitude * 0.1 * np.sin(4 * phase))
    else:
        wave = amplitude * base
    
    return wave * envelope

def generate_chord(frequencies, duration, sample_rate=SAMPLE_RATE, amplitude=0.2):
    """Generate a chord with multiple frequencies"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Add envelope
    envelope = np.ones_like(t)
    fade_samples = int(sample_rate * 0.01)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    # Combine all frequencies
    wave = np.zeros_like(t)
    for freq in frequencies:
        # Add harmonics to each note
        wave += amplitude * np.sin(2 * np.pi * freq * t)
        wave += amplitude * 0.2 * np.sin(2 * np.pi * freq * 2 * t)
    
    wave = wave * envelope
    return wave

def generate_noise_burst(duration, sample_rate=SAMPLE_RATE, amplitude=0.12):
    """Generate a short noise burst (for simple percussion) with quick envelope."""
    length = int(sample_rate * duration)
    if length <= 0:
        return np.array([], dtype=float)
    noise = np.random.uniform(-1.0, 1.0, length)
    # Fast attack/decay envelope for percussive feel
    env = np.ones(length, dtype=float)
    a = max(1, int(sample_rate * 0.002))  # 2ms attack
    d = max(1, int(sample_rate * max(0.0, duration - 0.004)))  # rest decay
    env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[a:a+d] = np.linspace(1, 0, d)
    return amplitude * noise * env

def line_to_music(line_properties):
    """Convert line properties to music with richer harmony and multiple parts"""
    if not line_properties:
        return np.zeros(SAMPLE_RATE * DURATION)
    
    points = line_properties['points']
    speed = line_properties['speed']
    height_range = line_properties['height_range']
    curvature = line_properties['curvature']
    
    # Determine tempo based on drawing speed
    min_note_duration = 0.15  # seconds
    max_note_duration = 0.6  # seconds
    
    # Normalize speed (assuming typical speed range of 100-1000 pixels/sec)
    normalized_speed = np.clip(speed / 500.0, 0.2, 2.0)
    base_note_duration = max_note_duration / normalized_speed
    
    # Segment the line into notes
    num_segments = int(DURATION / base_note_duration)
    num_segments = max(8, min(num_segments, 64))  # Between 8 and 64 notes
    
    segment_size = len(points) // num_segments
    if segment_size < 1:
        segment_size = 1
        num_segments = len(points)
    
    # Create multiple layers: melody, harmony, bass, arpeggio, pad, counter, perc
    total_length = SAMPLE_RATE * DURATION
    melody_track = np.zeros(total_length)
    harmony_track = np.zeros(total_length)
    bass_track = np.zeros(total_length)
    arpeggio_track = np.zeros(total_length)
    pad_track = np.zeros(total_length)
    counter_track = np.zeros(total_length)
    perc_track = np.zeros(total_length)
    
    # Chord progressions (indices into PENTATONIC_SCALE)
    # Create chord patterns based on curvature
    if curvature > 0.5:
        chord_pattern = [[0, 2, 4], [1, 3, 5], [2, 4, 0], [1, 3, 5]]  # More varied
    else:
        chord_pattern = [[0, 2, 4], [1, 3, 5], [0, 2, 4], [2, 4, 0]]  # More stable
    
    current_position = 0
    
    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = min((i + 1) * segment_size, len(points))
        
        if start_idx >= len(points):
            break
        
        # Get segment points
        segment_points = points[start_idx:end_idx]
        
        # Calculate average Y position for this segment (determines pitch)
        avg_y = sum(p[1] for p in segment_points) / len(segment_points)
        
        # Calculate X position for chord changes
        avg_x = sum(p[0] for p in segment_points) / len(segment_points)
        
        # Map Y position to scale note (inverted: top = high pitch, bottom = low pitch)
        note_index = int((1 - (avg_y - 50) / (CANVAS_HEIGHT - 50)) * (len(CURRENT_SCALE) - 1))
        note_index = max(0, min(note_index, len(CURRENT_SCALE) - 1))
        
        melody_freq = CURRENT_SCALE[note_index]
        # Counter-melody: pick a different scale degree (a third above)
        counter_index = (note_index + 2) % len(CURRENT_SCALE)
        counter_freq = CURRENT_SCALE[counter_index]
        
        # Determine chord based on X position (chord changes as you move horizontally)
        chord_index = int((avg_x / WIDTH) * len(chord_pattern)) % len(chord_pattern)
        chord_notes = chord_pattern[chord_index]
        
        # Get chord frequencies
        chord_freqs = [CURRENT_SCALE[n % len(CURRENT_SCALE)] for n in chord_notes]
        # Humanize: slight detune to avoid sterile chorusing
        if HUMANIZE_ENABLED:
            chord_freqs = [f * (1.0 + np.random.uniform(-0.003, 0.003)) for f in chord_freqs]
        
        # Bass note (root of the chord, one octave lower)
        bass_freq = chord_freqs[0] / 2
        
        # Calculate note duration
        if i < num_segments - 1:
            segment_time = segment_points[-1][2] - segment_points[0][2]
            note_duration = max(min_note_duration, min(segment_time * 2, max_note_duration))
        else:
            # Last note fills remaining time
            remaining_time = DURATION - (current_position / SAMPLE_RATE)
            note_duration = max(min_note_duration, remaining_time)

        # Humanize: apply jitter, swing, and articulation
        if HUMANIZE_ENABLED:
            # Jitter duration
            note_duration *= (1.0 + np.random.uniform(-HUMANIZE_INTENSITY, HUMANIZE_INTENSITY))
            note_duration = max(min_note_duration, min(note_duration, max_note_duration))
            # Swing on even/odd segments
            if SWING_FACTOR > 0:
                if i % 2 == 1:
                    note_duration *= (1.0 + SWING_FACTOR)
                else:
                    note_duration *= (1.0 - SWING_FACTOR)
            # Articulation (staccato/legato variation)
            articulation = 1.0 + np.random.uniform(-0.08, 0.05)
            note_duration = max(min_note_duration, min(note_duration * articulation, max_note_duration))
        
        # Generate tones with humanized amplitude and vibrato on melody
        m_amp = 0.22 * (1.0 + (np.random.uniform(-HUMANIZE_INTENSITY, HUMANIZE_INTENSITY) if HUMANIZE_ENABLED else 0.0))
        h_amp = 0.10 * (1.0 + (np.random.uniform(-0.1, 0.1) if HUMANIZE_ENABLED else 0.0))
        b_amp = 0.18 * (1.0 + (np.random.uniform(-0.08, 0.08) if HUMANIZE_ENABLED else 0.0))
        # Occasional melodic rest
        if HUMANIZE_ENABLED and np.random.rand() < 0.06:
            m_amp = 0.0
        vib_depth = 0.004 if HUMANIZE_ENABLED else 0.0
        attack_ms = 8 if HUMANIZE_ENABLED else 10
        release_ms = 14 if HUMANIZE_ENABLED else 10
        melody_tone = generate_tone(melody_freq, note_duration, amplitude=m_amp, harmonics=True,
                                    vibrato_depth=vib_depth, vibrato_rate=5.0, env_attack_ms=attack_ms, env_release_ms=release_ms)
        harmony_tone = generate_chord(chord_freqs, note_duration, amplitude=h_amp)
        bass_tone = generate_tone(bass_freq, note_duration, amplitude=b_amp, harmonics=False,
                                  vibrato_depth=0.0, vibrato_rate=0.0, env_attack_ms=12, env_release_ms=20)
        # Counter melody is slightly delayed for call-and-response feel
        counter_tone = generate_tone(counter_freq, note_duration * 0.9, amplitude=0.16, harmonics=True)
        
        # Arpeggio: split the segment into short sub-notes cycling through chord tones
        arp_steps = max(2, int(note_duration / 0.08))
        arp_unit = max(0.04, note_duration / arp_steps)
        arp_wave = np.array([], dtype=float)
        arp_order = chord_freqs + [melody_freq]
        for s in range(arp_steps):
            f = arp_order[s % len(arp_order)]
            arp_wave = np.concatenate([arp_wave, generate_tone(f, arp_unit, amplitude=0.08, harmonics=False)])
        # Ensure arpeggio matches segment length
        if len(arp_wave) > int(SAMPLE_RATE * note_duration):
            arp_wave = arp_wave[:int(SAMPLE_RATE * note_duration)]
        else:
            pad_len = int(SAMPLE_RATE * note_duration) - len(arp_wave)
            if pad_len > 0:
                arp_wave = np.concatenate([arp_wave, np.zeros(pad_len)])
        
        # Pad: sustained soft chord with slower attack/decay
        pad_len_sec = min(note_duration * 2.0, 2.0)
        pad_gain = CURRENT_STYLE_CONFIG.get('pad_gain', 1.0)
        pad_wave = generate_chord(chord_freqs, pad_len_sec, amplitude=0.06 * pad_gain)
        
        # Percussion: tiny noise burst at segment start; more intensity with curvature
        perc_base = 0.06 + float(np.clip(curvature, 0.0, 1.0)) * 0.06
        perc_gain = CURRENT_STYLE_CONFIG.get('perc_gain', 1.0)
        perc_amp = perc_base * perc_gain
        perc_wave = generate_noise_burst(min(0.04, note_duration), amplitude=perc_amp)
        
        # Add to tracks at current position
        end_position = current_position + len(melody_tone)
        
        if end_position > len(melody_track):
            end_position = len(melody_track)
            melody_tone = melody_tone[:end_position - current_position]
            harmony_tone = harmony_tone[:end_position - current_position]
            bass_tone = bass_tone[:end_position - current_position]
            counter_tone = counter_tone[:end_position - current_position]
            arp_wave = arp_wave[:end_position - current_position]
        
        melody_track[current_position:end_position] += melody_tone
        harmony_track[current_position:end_position] += harmony_tone
        bass_track[current_position:end_position] += bass_tone
        counter_offset = int(0.02 * SAMPLE_RATE)  # 20ms delay
        c_start = current_position + counter_offset
        c_end = min(c_start + len(counter_tone), len(counter_track))
        if c_start < len(counter_track):
            counter_track[c_start:c_end] += counter_tone[:max(0, c_end - c_start)]
        
        arpeggio_track[current_position:end_position] += arp_wave[:end_position - current_position]
        # Pad overlaps; place pad starting now but cap to buffer
        p_end = min(current_position + len(pad_wave), len(pad_track))
        pad_track[current_position:p_end] += pad_wave[:max(0, p_end - current_position)]
        # Percussion at the very start of segment
        per_end = min(current_position + len(perc_wave), len(perc_track))
        perc_track[current_position:per_end] += perc_wave[:max(0, per_end - current_position)]
        
        current_position = end_position
        
        # Stop if we've filled the duration
        if current_position >= len(melody_track):
            break
    
    # Mix all tracks together
    audio = (melody_track + harmony_track + bass_track +
             arpeggio_track + pad_track + counter_track + perc_track)
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.8
    
    return audio

def save_audio(audio_data, filename):
    """Save audio data as WAV file"""
    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio_data * 32767)
    
    filepath = os.path.join(MUSIC_DIR, filename)
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"音乐已保存为: {filepath}")

def play_audio(audio_data):
    """Play audio using pygame mixer"""
    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio_data * 32767)
    
    # Initialize pygame mixer
    pygame.mixer.quit()
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1, buffer=512)
    
    # Create sound object
    sound = pygame.mixer.Sound(audio_int16)
    sound.play()
    
    print("正在播放音乐...")

def main():
    global drawing, points, current_line, all_lines, line_start_time
    global music_generated, audio_data
    
    clock = pygame.time.Clock()
    running = True
    status_message = "Draw a line to create music!"
    message_time = 0
    
    while running:
        current_time = pygame.time.get_ticks() / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle button events
            for button in buttons:
                if button.handle_event(event):
                    if button == clear_button:
                        drawing_surface.fill(WHITE)
                        points = []
                        current_line = []
                        all_lines = []
                        music_generated = False
                        audio_data = None
                        status_message = "Board cleared"
                        message_time = current_time
                    
                    elif button == save_drawing_button:
                        if points:
                            filename = save_drawing_as_image()
                            status_message = f"Saved: {filename}"
                            message_time = current_time
                        else:
                            status_message = "No drawing to save"
                            message_time = current_time
                    
                    elif button == generate_music_button:
                        if points:
                            status_message = "Generating music..."
                            message_time = current_time
                            screen.blit(font.render(status_message, True, LIGHT_GREEN), (780, 30))
                            pygame.display.flip()
                            
                            line_props = calculate_line_properties(points)
                            audio_data = line_to_music(line_props)
                            music_generated = True
                            play_audio(audio_data)
                            status_message = "Music created and playing!"
                            message_time = current_time
                        else:
                            status_message = "Please draw a line first"
                            message_time = current_time
                    
                    elif button == style_button:
                        # Cycle style
                        global style_index, CURRENT_STYLE, CURRENT_STYLE_CONFIG, CURRENT_SCALE
                        style_index = (style_index + 1) % len(STYLE_NAMES)
                        CURRENT_STYLE = STYLE_NAMES[style_index]
                        CURRENT_STYLE_CONFIG = STYLES[CURRENT_STYLE]
                        CURRENT_SCALE = CURRENT_STYLE_CONFIG["scale"]
                        style_button.text = f"Style: {CURRENT_STYLE}"
                        status_message = f"Style set to {CURRENT_STYLE}"
                        message_time = current_time
                    
                    elif button == save_music_button:
                        if music_generated and audio_data is not None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"music_{timestamp}.wav"
                            save_audio(audio_data, filename)
                            status_message = f"Saved: {filename}"
                            message_time = current_time
                        else:
                            status_message = "Please generate music first"
                            message_time = current_time
            
            # Handle drawing
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_y = event.pos[1]
                if mouse_y >= CANVAS_TOP:  # Only draw in canvas area
                    drawing = True
                    line_start_time = current_time
                    current_line = []
                    points = []  # Clear previous line for single line drawing
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if drawing:
                    drawing = False
                    if current_line:
                        all_lines.append(current_line.copy())
            
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    mouse_x, mouse_y = event.pos
                    if mouse_y >= CANVAS_TOP:
                        adjusted_y = mouse_y - CANVAS_TOP
                        points.append((mouse_x, adjusted_y, current_time))
                        current_line.append((mouse_x, adjusted_y))
                        
                        # Draw on surface
                        if len(current_line) > 1:
                            pygame.draw.line(drawing_surface, BLACK, 
                                           current_line[-2], current_line[-1], 3)
        
        # Draw everything
        screen.fill(WHITE)
        
        # Draw top panel
        pygame.draw.rect(screen, GRAY, (0, 0, WIDTH, CANVAS_TOP))
        pygame.draw.line(screen, BLACK, (0, CANVAS_TOP), (WIDTH, CANVAS_TOP), 2)
        
        # Draw buttons
        for button in buttons:
            button.draw(screen)
        
        # Draw status message
        if current_time - message_time < 3:  # Show message for 3 seconds
            msg_surf = small_font.render(status_message, True, BLACK)
            screen.blit(msg_surf, (20, 75))
        else:
            instruction = small_font.render("Hold mouse to draw", True, DARK_GRAY)
            screen.blit(instruction, (20, 75))
        
        # Draw canvas
        screen.blit(drawing_surface, (0, CANVAS_TOP))
        
        # Info section removed - no longer displaying point count
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    print("=" * 50)
    print("线条转音乐程序")
    print("=" * 50)
    print("使用说明:")
    print("1. 在画板上按住鼠标绘制一条线")
    print("2. 点击'生成音乐'将线条转换为10秒音乐")
    print("3. 点击'保存绘画'保存你的绘画")
    print("4. 点击'保存音乐'保存生成的音乐")
    print("\n音乐生成规则:")
    print("- 绘画速度决定节奏快慢")
    print("- 线条高度决定音调高低（上方=高音，下方=低音）")
    print("- 使用五声音阶，保证音乐和谐")
    print("=" * 50)
    main()
