import atexit
import os
from pathlib import Path
import statistics
import subprocess
from time import sleep
from agents.mcp import MCPServerStdio, create_static_tool_filter
from openai import OpenAI
from openai.types import Reasoning
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
import agents
from synchronicity import Synchronizer
import geocoder


llama = subprocess.Popen([
    "llama-server",
    "-m",
    f"{Path().home()}/Models/Qwen3-0.6B-Q4_0.gguf",
    "--alias",
    "model",
    "--host",
    "127.0.0.1",
    "--port",
    "9931",
    "-c",
    "128000",
    "--chat-template-kwargs",
    '{"enable_thinking":false}',
])


def clean():
    llama.terminate()
    try:
        llama.wait(0.5)

    except subprocess.TimeoutExpired:
        llama.kill()


atexit.register(clean)

agents.set_default_openai_api("chat_completions")
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9931/v1"

agent = agents.Agent(
    "worker",
    instructions="You are a smart home assistant who is running through voice. Use the available tools when needed to provide an accurate response. Respond in no more than 3 sentences.",
    model="model",
    model_settings=agents.ModelSettings(tool_choice="required"),
)

sleep(2)
print(OpenAI(base_url="http://127.0.0.1:9931/v1").models.list())

syncer = Synchronizer()


@syncer.create_blocking
async def run_agent(text, server):
    async with server:
        agent.mcp_servers = [server]
        response = agents.Runner.run_streamed(
            starting_agent=agent,
            input=text,
        )

        async for event in response.stream_events():
            yield event


word = OpenWakeWord.from_builtin(Model.HEY_JARVIS)
features = OpenWakeWordFeatures.from_builtin()

transcriber = pywhispercpp.model.Model("base.en")
router = Router()
model: OnnxWrapper = silero_vad.load_silero_vad(onnx=True)  # pyright: ignore[reportAssignmentType]
tts_model = TTSModel.load_model()
voice_state = tts_model.get_state_for_audio_prompt(
    "hf://kyutai/tts-voices/alba-mackenna/casual.wav"  # ty:ignore[invalid-argument-type]
)

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
                    print(choice, route)
                    if route["answers"]["department"]["answer_confidence"] < 0.65:
                        choice = "handoff to llm"

                    elif (
                        0.65 < route["answers"]["department"]["answer_confidence"] < 0.8
                    ):
                        confidence: list[float] = [
                            route["answers"]["department"]["answer_confidence"]
                        ]
                        confidence.extend([
                            router.predict(text, questions)["answers"]["department"][
                                "answer_confidence"
                            ]
                            for i in range(6)
                        ])
                        if (
                            statistics.fmean(confidence) < 0.7
                            or not len(set[float](confidence)) == 1
                        ):
                            choice = "handoff to llm"

                    if choice == "temperature":
                        server = MCPServerStdio(
                            {
                                "command": "open-meteo-mcp-server",
                            },
                            tool_filter=create_static_tool_filter([
                                "weather_forecast",
                                "weather_archive",
                                "geocoding",
                            ]),
                        )

                        addon = f"COORDINATES: {geocoder.ip('me').latlng}. its longitude, latitude. Give me Fahrenheit. In your response, don't use coordinates, geocode them to a town. and say, town, state. dont get more specific."

                    else:
                        answer = "Sorry, this is unsupported"
                        addon = ""
                    print("playing")
                    for chunk in run_agent(text + f"  {addon}", server):  # pyright: ignore[reportPossiblyUnboundVariable, reportCallIssue, reportGeneralTypeIssues]
                        print(chunk, end="", flush=True)
                    audio = tts_model.generate_audio_stream(voice_state, answer)  # ty:ignore[too-many-positional-arguments]
                    ostream = sd.OutputStream(24000, channels=1, dtype="float32")
                    ostream.start()
                    for chunk in audio:
                        ostream.write(chunk)

                    ostream.close()
                    word.reset()
                    features.reset()
