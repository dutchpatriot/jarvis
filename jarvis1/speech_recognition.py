import sounddevice as sd
import queue
import time
import numpy as np
import noisereduce as nr
from config import (SILENCE_SKIP_DB, SPEECH_START_DB, SILENCE_DROP_DB, SILENCE_DURATION,
                    SILENCE_DURATION_EXT, WHISPER_MODEL, WHISPER_BEAM_SIZE,
                    WHISPER_SAMPLE_RATE, STT_BLOCKSIZE, USE_GPU, GPU_DEVICE_ID,
                    WHISPER_COMPUTE_TYPE, NOISE_REDUCE)
from colors import stt

# Lazy load - deferred to avoid startup delay
_device = None
_device_index = 0
_compute_type = None
_whisper_model = None
_WhisperModel = None

def _select_best_gpu():
    """Auto-select GPU with most VRAM, or use configured GPU_DEVICE_ID."""
    import torch

    if GPU_DEVICE_ID is not None:
        return GPU_DEVICE_ID

    # Auto-select: pick GPU with most VRAM
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return 0
    if num_gpus == 1:
        return 0

    # Multiple GPUs: select by VRAM
    best_gpu = 0
    best_vram = 0
    for i in range(num_gpus):
        vram = torch.cuda.get_device_properties(i).total_memory
        name = torch.cuda.get_device_name(i)
        print(stt(f"  GPU {i}: {name} ({vram // (1024**3)}GB)"))
        if vram > best_vram:
            best_vram = vram
            best_gpu = i

    return best_gpu


def get_device():
    """Detect best available device (GPU with CPU fallback)."""
    global _device, _device_index, _compute_type
    if _device is None:
        if USE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_id = _select_best_gpu()
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory // (1024**3)
                    print(stt(f"Using CUDA: {gpu_name} ({gpu_mem}GB)"))
                    _device = "cuda"
                    _device_index = gpu_id
                    _compute_type = WHISPER_COMPUTE_TYPE
                else:
                    print(stt("GPU requested but CUDA not available - using CPU"))
                    _device, _device_index, _compute_type = "cpu", 0, "int8"
            except ImportError:
                print(stt("GPU requested but torch not found - using CPU"))
                _device, _device_index, _compute_type = "cpu", 0, "int8"
            except Exception as e:
                print(stt(f"GPU init failed ({e}) - using CPU"))
                _device, _device_index, _compute_type = "cpu", 0, "int8"
        else:
            print(stt("Using CPU (GPU disabled in config)"))
            _device, _device_index, _compute_type = "cpu", 0, "int8"
    return _device, _device_index, _compute_type

def init_whisper(model_size=None):
    """Initialize Whisper model. Sizes: tiny, base, small, medium, large-v2"""
    global _whisper_model, _WhisperModel
    if model_size is None:
        model_size = WHISPER_MODEL
    if _whisper_model is None:
        # Lazy import
        if _WhisperModel is None:
            print(stt("Loading faster-whisper library..."))
            from faster_whisper import WhisperModel
            _WhisperModel = WhisperModel

        device, device_index, compute_type = get_device()
        print(stt(f"Loading Whisper model '{model_size}' on {device} (GPU {device_index})..."))
        _whisper_model = _WhisperModel(model_size, device=device, device_index=device_index, compute_type=compute_type)
        print(stt("Whisper ready!"))
    return _whisper_model

def list_microphones():
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            dev = dict(device)
            dev['index'] = i
            input_devices.append(dev)
    for i, device in enumerate(input_devices):
        print(f"{i}: {device['name']} (device {device['index']})")
    return input_devices

def get_default_microphone(input_devices):
    """Get first available microphone without prompting."""
    selected_device = input_devices[0]
    samplerate = selected_device['default_samplerate']
    print(f"Using microphone: {selected_device['name']} ({samplerate} Hz)")
    return selected_device, samplerate

def select_microphone_and_samplerate(input_devices):
    choice = int(input("Select the microphone by entering the corresponding number: "))
    selected_device = input_devices[choice]
    samplerate = selected_device['default_samplerate']
    print(f"Selected microphone: {selected_device['name']}")
    print(f"Sample rate: {samplerate} Hz")
    return selected_device, samplerate

def whisper_speech_to_text(selected_device, samplerate, extended_listen=False):
    """Record audio and transcribe with Whisper.

    Args:
        extended_listen: If True, waits for longer pause before transcribing.
    """
    model = init_whisper()
    q = queue.Queue()
    audio_buffer = []

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata.copy())

    try:
        with sd.InputStream(samplerate=int(samplerate), blocksize=STT_BLOCKSIZE,
                           device=selected_device['index'], dtype='float32',
                           channels=1, callback=callback):

            if extended_listen:
                print("# Speak your question... (pause to finish)")
            else:
                print("# Say something!")

            peak_db = -60.0
            drop_start = None
            speech_started = False
            silence_threshold = SILENCE_DURATION_EXT if extended_listen else SILENCE_DURATION

            while True:
                try:
                    data = q.get(timeout=0.3)
                except queue.Empty:
                    continue

                audio_buffer.append(data)

                # Calculate dB
                rms = np.sqrt(np.mean(data ** 2)) if len(data) > 0 else 0
                current_db = 20 * np.log10(rms) if rms > 1e-10 else -60.0

                # Track peak (ignore clipped)
                if current_db > peak_db and current_db < -5:
                    peak_db = current_db
                    drop_start = None
                    if current_db > SPEECH_START_DB:
                        speech_started = True

                # Show dB meter
                bar_len = int((current_db + 60) / 60 * 20)
                bar = '█' * max(0, min(20, bar_len)) + '░' * (20 - max(0, min(20, bar_len)))
                print(f"\r[{bar}] {current_db:5.1f}dB ", end='', flush=True)

                # Check for silence after speech
                if speech_started and current_db < (peak_db - SILENCE_DROP_DB):
                    if drop_start is None:
                        drop_start = time.time()
                    elif (time.time() - drop_start) > silence_threshold:
                        print()  # Newline after meter
                        break
                else:
                    drop_start = None

        # Convert buffer to numpy array
        if not audio_buffer:
            return ""

        audio_data = np.concatenate(audio_buffer, axis=0).flatten()

        # Check if there was actual audio (not just silence)
        rms = np.sqrt(np.mean(audio_data ** 2))
        avg_db = 20 * np.log10(rms) if rms > 1e-10 else -60.0
        if avg_db < SILENCE_SKIP_DB:  # Too quiet, probably no speech
            print(stt(f"Skipping - too quiet ({avg_db:.1f}dB)"))
            return ""

        # Transcribe directly from memory (no temp file needed)
        # faster-whisper accepts numpy array - resample to 16kHz if needed
        if int(samplerate) != WHISPER_SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(audio_data) * WHISPER_SAMPLE_RATE / samplerate)
            audio_16k = signal.resample(audio_data, num_samples).astype(np.float32)
        else:
            audio_16k = audio_data.astype(np.float32)

        # Apply noise reduction if enabled
        if NOISE_REDUCE:
            print(stt("Reducing noise..."))
            audio_16k = nr.reduce_noise(y=audio_16k, sr=WHISPER_SAMPLE_RATE, prop_decrease=0.8)

        print(stt("Transcribing..."))
        segments, info = model.transcribe(
            audio_16k,
            beam_size=WHISPER_BEAM_SIZE,
            # Anti-hallucination settings
            no_speech_threshold=0.6,           # Skip if probability of no speech > 60%
            log_prob_threshold=-1.0,           # Skip low confidence segments
            hallucination_silence_threshold=0.5,  # Skip hallucinations after 0.5s silence
            condition_on_previous_text=False,  # Don't let previous text influence (reduces repetition)
        )
        text = " ".join([seg.text for seg in segments]).strip()

        # Filter non-Latin hallucinations (Hindi, Chinese, Arabic, etc.)
        import re
        # Remove leading non-ASCII characters and clean up
        text = re.sub(r'^[^\x00-\x7F]+\s*', '', text)
        # Remove any remaining non-Latin script blocks
        text = re.sub(r'[\u0900-\u097F]+', '', text)  # Devanagari (Hindi)
        text = re.sub(r'[\u4E00-\u9FFF]+', '', text)  # Chinese
        text = re.sub(r'[\u0600-\u06FF]+', '', text)  # Arabic
        text = re.sub(r'[\u0400-\u04FF]+', '', text)  # Cyrillic
        text = re.sub(r'[\u3040-\u30FF]+', '', text)  # Japanese
        text = re.sub(r'[\uAC00-\uD7AF]+', '', text)  # Korean
        text = re.sub(r'\s+', ' ', text).strip()

        print(stt(f"Result: {text}"))
        return text

    except Exception as e:
        print(f"\nAn error occurred during audio processing: {e}")
        # Try to free GPU memory on OOM errors
        if "out of memory" in str(e).lower():
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(stt("Cleared GPU cache after OOM"))
            except:
                pass
    return ""

