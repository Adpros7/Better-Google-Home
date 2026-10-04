from pathlib import Path

from pocket_tts import TTSModel, export_model_state

if not Path("my_voice.wav").exists():
    import record  # noqa: F401

model = TTSModel.load_model()
voice_state = model.get_state_for_audio_prompt(Path("my_voice.wav"))  # ty:ignore[invalid-argument-type]
export_model_state(voice_state, "my_voice.safetensors")