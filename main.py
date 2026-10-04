import atexit
import os
import subprocess
from pocket_tts import TTSModel
import numpy as np
import pywhispercpp.model
import silero_vad
import sounddevice as sd
import stopwatch
import torch
from pyopen_wakeword import Model, OpenWakeWord, OpenWakeWordFeatures
from silero_vad.utils_vad import OnnxWrapper
from laya import Router
from questions import questions
from temperature import get_temperature
import agents

proc = subprocess.Popen([
    "llama-server",
    "-m",
    "~/Models/Qwen3-0.6B-Q4_0.gguf",
    "--host",
    "127.0.0.1",
    "--port",
    "9931",
    "-c",
    "4096",
])


def clean():
    proc.terminate()
    try:
        proc.wait(8)

    except subprocess.TimeoutExpired:
        proc.kill()


atexit.register(clean)

agents.set_default_openai_api("chat_completions")
os.environ["OPENAI_BASE_URL"] = "127.0.0.1:9931/v1"

agent = agents.Agent(
    "worker",
    instructions="You are a samrt home assistant who is running through voice. Use the available tools when needed to provide an accurate response. Respond in no more than 3 sentences.",
)

word = OpenWakeWord.from_builtin(Model.HEY_JARVIS)
features = OpenWakeWordFeatures.from_builtin()

transcriber = pywhispercpp.model.Model("base.en")
router = Router()
model: OnnxWrapper = silero_vad.load_silero_vad(onnx=True)  # pyright: ignore[reportAssignmentType]
tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt(
    "hf://kyutai/tts-voices/alba-mackenna/casual.wav"
)  # ty:ignore[invalid-argument-type]

with sd.RawInputStream(16000, channels=1, dtype="int16", blocksize=1280) as stream:
    while True:
        data, overflowed = stream.read(1280)
        for embedding in features.process_streaming(bytes(data)):
            for probability in word.process_streaming(embedding):
                if probability > 0.5:
                    print("Hey Jarvis")
                    with sd.RawInputStream(
                        samplerate=16000, blocksize=512, channels=1, dtype="int16"
                    ) as talk:
                        watch = stopwatch.Stopwatch()
                        watch.start()
                        recording = []
                        while True:
                            data, overflowed = talk.read(512)
                            audio = torch.from_numpy(
                                np.frombuffer(data, dtype=np.int16).astype(np.float32)
                                / 32768.0
                            )
                            recording.append(audio)
                            speech_probability = model(audio, 16000).item()
                            if speech_probability > 0.5:
                                print("speech")
                                watch.reset()
                                watch.start()

                            elif watch.elapsed > 1.3:
                                break

                    audio = torch.cat(recording).numpy()

                    segments = transcriber.transcribe(audio)
                    text = "".join(segment.text for segment in segments)
                    print(text)
                    route = router.predict(text, questions)
                    choice = route["answers"]["department"]["choice"]
                    print(choice)
                    if choice == "temperature":
                        latitude, longitude = eval(os.environ["MY_LOCATION"])
                        temp, feels_like_temp = get_temperature(latitude, longitude)
                        answer = f"The temperature is {temp} degrees Fahrenheit and the feels tike temperature is {feels_like_temp} degrees Fahrenheit"

                    print("playing")
                    audio = tts_model.generate_audio_stream(voice_state, answer)  # ty:ignore[too-many-positional-arguments]
                    ostream = sd.OutputStream(24000, channels=1, dtype="float32")
                    ostream.start()
                    for chunk in audio:
                        ostream.write(chunk)

                    ostream.close()
                    word.reset()
                    features.reset()
