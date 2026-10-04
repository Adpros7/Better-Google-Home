import pyaudio
import wave

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono
RATE = 44100  # Sample rate
RECORD_SECONDS = 15
WAVE_OUTPUT_FILENAME = "my_voice.wav"

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

print("* recording")

# Record audio
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

# Stop and close stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the file
wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()
