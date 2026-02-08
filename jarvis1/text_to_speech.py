import os
import time
import numpy as np
import sounddevice as sd
import re

# ONNX Runtime optimizations for Tensor Cores (RTX 3070)
os.environ.setdefault('ORT_TENSORRT_FP16_ENABLE', '1')  # Enable FP16 in TensorRT
os.environ.setdefault('ORT_TENSORRT_ENGINE_CACHE_ENABLE', '1')  # Cache compiled engines
os.environ.setdefault('ORT_TENSORRT_ENGINE_CACHE_PATH', os.path.expanduser('~/.cache/onnx_tensorrt'))

from config import (INTERRUPT_DB, INTERRUPT_DURATION,
                    TTS_SPEED_EN, TTS_PITCH_EN, TTS_VOLUME_EN,
                    TTS_SPEED_NL, TTS_PITCH_NL, TTS_VOLUME_NL,
                    TTS_GRACE_PERIOD, TTS_LANG_THRESHOLD, TTS_LOG_LENGTH,
                    AUDIO_SAMPLE_RATE, AUDIO_BLOCKSIZE, USE_GPU, GPU_DEVICE_ID,
                    FORCE_LANGUAGE)
from colors import tts as tts_log

# Piper TTS - lazy load for fast startup
_piper_voice_nl = None
_piper_voice_en = None
VOICE_DIR = os.path.expanduser("~/.local/share/piper/voices")

# Runtime language override (None = auto, "en" = English, "nl" = Dutch)
_forced_language = FORCE_LANGUAGE


def set_language(lang):
    """Set forced language: 'en', 'nl', or None for auto-detect."""
    global _forced_language
    _forced_language = lang
    if lang is None:
        print(tts_log("Language: auto-detect"))
    else:
        print(tts_log(f"Language: forced to {lang}"))


def get_language():
    """Get current forced language setting."""
    return _forced_language


# Selected GPU ID (cached)
_selected_gpu_id = None

def _select_best_gpu():
    """Auto-select GPU with most VRAM, or use configured GPU_DEVICE_ID."""
    global _selected_gpu_id
    if _selected_gpu_id is not None:
        return _selected_gpu_id

    import torch

    if GPU_DEVICE_ID is not None:
        _selected_gpu_id = GPU_DEVICE_ID
        return _selected_gpu_id

    # Auto-select: pick GPU with most VRAM
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        _selected_gpu_id = 0
        return 0

    # Multiple GPUs: select by VRAM
    best_gpu = 0
    best_vram = 0
    for i in range(num_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory
        if vram > best_vram:
            best_vram = vram
            best_gpu = i

    _selected_gpu_id = best_gpu
    return best_gpu


def _check_cuda_available():
    """Check if CUDA is available for Piper."""
    if not USE_GPU:
        return False
    try:
        import torch
        if torch.cuda.is_available():
            # Set the selected GPU as default
            gpu_id = _select_best_gpu()
            torch.cuda.set_device(gpu_id)

            # Enable TensorRT for Tensor Cores if available
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'TensorrtExecutionProvider' in providers:
                    print(tts_log("TensorRT available - Tensor Cores enabled"))
                elif 'CUDAExecutionProvider' in providers:
                    print(tts_log("CUDA available - using GPU"))
            except ImportError:
                pass

            return True
        return False
    except ImportError:
        return False

def _get_voice(lang="nl"):
    """Lazy load Piper voice (GPU if available, CPU fallback)."""
    global _piper_voice_nl, _piper_voice_en

    use_cuda = _check_cuda_available()

    if lang == "nl":
        if _piper_voice_nl is None:
            from piper import PiperVoice
            model_path = os.path.join(VOICE_DIR, "nl_BE-nathalie-medium.onnx")
            device_str = "GPU" if use_cuda else "CPU"
            print(tts_log(f"Loading Dutch voice ({device_str})..."))
            _piper_voice_nl = PiperVoice.load(model_path, use_cuda=use_cuda)
            print(tts_log("Dutch voice ready"))
        return _piper_voice_nl
    else:
        if _piper_voice_en is None:
            from piper import PiperVoice
            model_path = os.path.join(VOICE_DIR, "en_US-lessac-medium.onnx")
            device_str = "GPU" if use_cuda else "CPU"
            print(tts_log(f"Loading English voice ({device_str})..."))
            _piper_voice_en = PiperVoice.load(model_path, use_cuda=use_cuda)
            print(tts_log("English voice ready"))
        return _piper_voice_en


def detect_language(text):
    """Simple language detection based on common words."""
    nl_words = ["de", "het", "een", "van", "en", "in", "is", "dat", "op", "te",
                "voor", "met", "zijn", "niet", "aan", "dit", "ook", "als", "maar", "om",
                "je", "ik", "we", "hij", "zij", "u", "kan", "zou", "wel", "nog"]

    words = text.lower().split()
    if len(words) < 3:
        # For short phrases, check if any word is distinctly Dutch
        for w in words:
            if w in nl_words and w not in ["de", "is", "in", "en", "van"]:  # Skip words common in both
                return "nl"
        return "en"  # Default to English for short phrases

    nl_count = sum(1 for w in words if w in nl_words)
    ratio = nl_count / len(words)

    return "nl" if ratio > TTS_LANG_THRESHOLD else "en"


def clean_text(text):
    """Replace unsupported characters with speakable alternatives."""
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Replace special characters
    replacements = {
        '#': ' hashtag ', '@': ' at ', '&': ' and ', '%': ' percent ',
        '$': ' dollar ', '*': '', '+': ' plus ', '=': ' equals ',
        '<': ' less than ', '>': ' greater than ', '/': ' slash ',
        '\\': ' backslash ', '|': ' pipe ', '~': ' tilde ', '^': ' caret ',
        '_': ' ', '{': '', '}': '', '[': '', ']': '', '`': '',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove emojis and non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def speak(text, speed=None, pitch=None, volume=None, interruptable=True, lang=None):
    """Speak text using Piper TTS with optional interrupt detection.

    Args:
        text: Text to speak
        speed: Speech rate override (None = use per-language config)
        pitch: Pitch override (None = use per-language config)
        volume: Volume multiplier override (None = use per-language config)
        interruptable: If True, monitor mic for interrupt
        lang: Force language ('nl' or 'en'), or None for auto-detect

    Returns:
        True if interrupted, False otherwise
    """
    text = clean_text(text)

    if not text or text.isspace():
        return False

    # Use forced language if set, otherwise auto-detect
    if lang is None:
        if _forced_language is not None:
            lang = _forced_language
            print(tts_log(f"[DEBUG] Using forced lang: {lang}"))
        else:
            lang = detect_language(text)
            print(tts_log(f"[DEBUG] Auto-detected lang: {lang} for: {text[:30]}..."))

    # Apply per-language settings (can be overridden by explicit params)
    if lang == "nl":
        speed = speed if speed is not None else TTS_SPEED_NL
        pitch = pitch if pitch is not None else TTS_PITCH_NL
        volume = volume if volume is not None else TTS_VOLUME_NL
    else:
        speed = speed if speed is not None else TTS_SPEED_EN
        pitch = pitch if pitch is not None else TTS_PITCH_EN
        volume = volume if volume is not None else TTS_VOLUME_EN

    if TTS_LOG_LENGTH > 0 and len(text) > TTS_LOG_LENGTH:
        print(tts_log(f"Speaking ({lang}): {text[:TTS_LOG_LENGTH]}..."))
    else:
        print(tts_log(f"Speaking ({lang}): {text}"))

    try:
        voice = _get_voice(lang)

        # Synthesize - returns AudioChunk objects
        chunks = list(voice.synthesize(text))
        if not chunks:
            return False

        # Combine all audio chunks
        audio_arrays = [chunk.audio_float_array for chunk in chunks]
        audio_float = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
        sample_rate = chunks[0].sample_rate

        # Apply pitch adjustment (resample to change pitch, then adjust playback rate)
        # pitch < 1.0 = lower pitch, pitch > 1.0 = higher pitch
        if pitch != 1.0:
            from scipy import signal
            # Resample to change pitch - more samples = lower pitch when played at same rate
            new_length = int(len(audio_float) / pitch)
            audio_float = signal.resample(audio_float, new_length)
            # Adjust sample rate to maintain original duration
            sample_rate = int(sample_rate / pitch)

        # Apply speed adjustment if needed
        if speed != 1.0:
            from scipy import signal
            new_length = int(len(audio_float) / speed)
            audio_float = signal.resample(audio_float, new_length)

        # Apply volume adjustment
        if volume != 1.0:
            audio_float = audio_float * volume
            # Clip to prevent distortion
            audio_float = np.clip(audio_float, -1.0, 1.0)

        if not interruptable:
            # Simple playback
            sd.play(audio_float, sample_rate)
            sd.wait()
            return False

        # Interruptable playback with mic monitoring
        # Play audio in background
        sd.play(audio_float, sample_rate)

        start_time = time.time()
        loud_start = None

        try:
            with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype='float32', blocksize=AUDIO_BLOCKSIZE) as stream:
                while sd.get_stream().active:
                    audio, _ = stream.read(AUDIO_BLOCKSIZE)

                    # Grace period - ignore first TTS_GRACE_PERIOD seconds
                    if time.time() - start_time < TTS_GRACE_PERIOD:
                        continue

                    rms = np.sqrt(np.mean(audio ** 2))
                    db = 20 * np.log10(rms) if rms > 1e-10 else -60.0

                    # Track sustained loud audio
                    if db > INTERRUPT_DB:
                        if loud_start is None:
                            loud_start = time.time()
                        elif time.time() - loud_start > INTERRUPT_DURATION:
                            print(f"\n{tts_log(f'Break detected! (sustained {db:.1f}dB)')}")
                            sd.stop()
                            time.sleep(0.2)
                            return True
                    else:
                        loud_start = None
        except Exception as e:
            sd.wait()

        return False

    except Exception as e:
        print(tts_log(f"TTS error: {e}"))
        return False
