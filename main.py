import atexit
from ipaddress import ip_address
from json import load
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import sleep
from openai import APIConnectionError, InternalServerError, OpenAI
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
from playsound3 import playsound
from fabric import Connection
from dotenv import load_dotenv

load_dotenv(".env.local")

startLlamaServerCmd = ["llama-server",
    
    "--hf-repo",
    "mradermacher/hua-1.7b-GGUF",
    "--hf-file",
    "hua-1.7b.Q4_K_M.gguf",
    "--alias",
    "model",
    "--host",
    "0.0.0.0",
    "--port",
    "9931",
    "-c",
    "8192",
    "--reasoning",
    "off",
    "--jinja",
    "--chat-template-kwargs",
    '{"enable_thinking":false}',]
    
MAIN_MACHINE = "darwin"
ssh = False
if not sys.platform == MAIN_MACHINE:
    ssh = True
    connection = Connection(os.environ["HOST_ADDRESS"], os.environ["HOST_USER"], connect_kwargs={"password": os.environ["PASSWORD"]})
    sshPID = connection.run(startLlamaServerCmd, asynchronous=True).pid

if not ssh:
    llama = subprocess.Popen(startLlamaServerCmd) if no


def clean():
    if ssh:
        connection.run(f"kill -9 {sshPID}")
        return
        
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

    except (InternalServerError, APIConnectionError):
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
                    
                    word.reset()
                    features.reset()
