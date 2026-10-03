from pyopen_wakeword.const import Model


from silero_vad.sequence_vad import SileroVADSequence


from silero_vad.utils_vad import OnnxWrapper


from typing import Any

import stopwatch


import time

import numpy as np
import silero_vad
import sounddevice as sd
import torch
from pyopen_wakeword import OpenWakeWord, OpenWakeWordFeatures, Model
import pywhispercpp.model

word = OpenWakeWord.from_builtin(Model.HEY_JARVIS)
features = OpenWakeWordFeatures.from_builtin()

transcriber = pywhispercpp.model.Model("base.en")
model: OnnxWrapper = silero_vad.load_silero_vad(onnx=True)  # pyright: ignore[reportAssignmentType]

with sd.RawInputStream(16000, channels=1, dtype="int16", blocksize=1280) as stream:
    while True:
        data, overflowed = stream.read(1280)
        for embedding in features.process_streaming(bytes(data)):
            for probability in word.process_streaming(embedding):

                if probability > 0.5:
                    print("Hey Jarvis")
                    with sd.RawInputStream(samplerate=16000, blocksize=512, channels=1, dtype="int16") as talk:
                        watch = stopwatch.Stopwatch()
                        watch.start()
                        recording = []
                        while True:
                            data, overflowed = talk.read(512)
                            audio = torch.from_numpy(
                                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
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
                    word.reset()
                    features.reset()

                    
                

