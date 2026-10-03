from silero_vad.sequence_vad import SileroVADSequence


from silero_vad.utils_vad import OnnxWrapper


from typing import Any


import time

import numpy as np
import silero_vad
import sounddevice as sd
import torch
from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures, Model

word = OpenWakeWord.from_builtin(Model.HEY_JARVIS)
features = OpenWakeWordFeatures.from_builtin()

model: OnnxWrapper = silero_vad.load_silero_vad(onnx=True)  # pyright: ignore[reportAssignmentType]

with sd.RawInputStream(16000, channels=1, dtype="int16", blocksize=1280) as stream:
    while True:
        data, overflowed = stream.read(1280)
        for embedding in features.process_streaming(bytes(data)):
            for probability in word.process_streaming(embedding):

                if probability > 0.5:
                    print("Hey Jarvis")
                    with sd.RawInputStream(samplerate=16000, blocksize=512, channels=1, dtype="int16") as talk:
                        while True:
                            data, overflowed = stream.read(512)
                            audio = torch.from_numpy(
                                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                            )
                            speech_probability = model(audio, 16000).item()
                            if speech_probability > 0.5:
                                print("speech")
                    
                

