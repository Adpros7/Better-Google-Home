import atexit
import os
import statistics
import subprocess
from time import sleep
from xsmeteo.models.forecast import ForecastResponse
from geocoder import ip
from openai import InternalServerError, OpenAI
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
from geopy.geocoders import Nominatim
from xsmeteo import XSMeteo
from playsound3 import playsound

llama = subprocess.Popen([
    "llama-server",
    "--hf-repo",
    "mradermacher/hua-1.7b-GGUF",
    "--hf-file",
    "hua-1.7b.Q4_K_M.gguf",
    "--alias",
    "model",
    "--host",
    "127.0.0.1",
    "--port",
    "9931",
    "-c",
    "8192",
    "--reasoning",
    "off",
    "--jinja",
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
    instructions="You are a smart home assistant who is running through voice. Use the available tools when needed to provide an accurate response. Respond in no more than 3 sentences. NO EMOJIS. ALWAYS WRITE OUT THE SYMBOLS AND NUMBERS AS WORDS. so for example if you want to write $50, write fifty dollars",
    model="model",
    model_settings=agents.ModelSettings(max_tokens=512),
)


while True:
    try:
        OpenAI(base_url="http://127.0.0.1:9931/v1").models.list()
        break
    
    except InternalServerError:
        sleep(0.5)

syncer = Synchronizer()


@syncer.create_blocking
async def run_agent(text, server=None):
    if server:
        async with server:
            agent.mcp_servers = [server]
            response = agents.Runner.run_streamed(
                starting_agent=agent,
                input=text,
            )
    else:
        response = agents.Runner.run_streamed(starting_agent=agent, input=text)

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
                    sound = playsound("wake-tone.mp3", block=False)
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
                                sound.stop()
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

                    answer = ""
                    if choice == "temperature":

                        lat, long = eval(os.environ["MY_LOCATION"]) # in the form "(lat, long)""
                        with XSMeteo() as client:
                            current: ForecastResponse = client.get_forecast(
                                latitude=lat,
                                longitude=long,
                                current=["temperature_2m", "apparent_temperature"],
                                temperature_unit="fahrenheit",
                            )
                            print(current.current)
                        addon = f"SYSTEM PROMPT: Take this information and take what you need to ANSWER the question comprehensively yet concisely LOCATION: {Nominatim(user_agent='betterghome').reverse(ip('me').latlng)}, \
                            CURRENT TEMPERATURE: {round(float(current.current['temperature_2m']))}, \
                            FEELS LIKE TEMPERATURE: {round(float(current.current['apparent_temperature']))}\
                            FORMATTING RULES: NO NEWLINES OR ANYTHING. ONLY GIVE THE USER WHAT THEY ASKED FOR. NO EMOJIS. ALWAYS WRITE OUT THE SYMBOLS AND NUMBERS AS WORDS. for example 45.4°F becomes forty-five point four degrees fahrenheit. also, if referencing a location, you only need the town and place. so Grove street, watertown, MA, USA ebcomes watertown, Massachusetts\
                            Also, remember to only give what the user asked for, if they asked for temperatature, give them ONLY temperature, not also feels like temperature. An for god sakes, don't be too specfc with the location. Also, try to sound natural. For example, say things like It's twenty-four degrees fahrenheit in Watertown, Massachusetts. DON'T say its 24°F"

                    else:
                        answer = "Sorry, this is unsupported"
                        addon = ""
                    print("playing")
                    for chunk in run_agent(text + f"  {addon}"):  # pyright: ignore[reportPossiblyUnboundVariable, reportCallIssue, reportGeneralTypeIssues]
                        print(chunk, end="", flush=True)
                        if (
                            chunk.type == "run_item_stream_event"
                            and chunk.name == "message_output_created"
                        ):
                            answer += agents.ItemHelpers.text_message_output(chunk.item)

                    print(answer)
                    audio = tts_model.generate_audio_stream(voice_state, answer)  # ty:ignore[too-many-positional-arguments]
                    ostream = sd.OutputStream(24000, channels=1, dtype="float32")
                    ostream.start()
                    for chunk in audio:
                        ostream.write(chunk)

                    ostream.close()
                    word.reset()
                    features.reset()
