import pyaudio
import wave
import librosa

# Parameters for recording
duration = 5  # Record for 5 seconds
sample_rate = 22050  # Sample rate
channels = 1  # Mono audio
filename = "test_recording.wav"  # Output file name

# Initialize PyAudio
p = pyaudio.PyAudio()

# Step 1: Record Audio using PyAudio and Save to WAV file
print("Recording...")
frames = []

# Open stream for recording
stream = p.open(format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024)

for _ in range(0, int(sample_rate / 1024 * duration)):
    data = stream.read(1024)
    frames.append(data)

print("Recording finished.")

# Stop and close the stream
stream.stop_stream()
stream.close()

# Step 2: Save the audio to a .wav file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(sample_rate)
wf.writeframes(b''.join(frames))
wf.close()

# Step 3: Playback the recorded audio
import sounddevice as sd
audio_data, sr = librosa.load(filename, sr=sample_rate, mono=True)  # Load the saved audio file
sd.play(audio_data, sr)
sd.wait()
