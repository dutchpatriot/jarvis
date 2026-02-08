import sounddevice as sd
import numpy as np
from openwakeword.model import Model
from scipy import signal
import time
from colors import wake
from config import WAKE_WORD, WAKE_THRESHOLD, AUDIO_SAMPLE_RATE

# Global wake word model
_oww_model = None

def init_wake_word():
    """Initialize OpenWakeWord model."""
    global _oww_model
    if _oww_model is None:
        print(wake("Loading wake word model..."))
        # Load all default wake word models (includes hey_jarvis, alexa, etc.)
        _oww_model = Model()
        print(wake(f"Ready! Say '{WAKE_WORD.replace('_', ' ').title()}' to activate."))
    return _oww_model

def listen_for_wake_word(selected_device, samplerate, timeout=None):
    """
    Efficient wake word detection - uses minimal CPU while waiting.
    Returns True when wake word detected, False on timeout, None on error.
    """
    model = init_wake_word()

    # OpenWakeWord expects 16kHz, resample if needed
    target_rate = AUDIO_SAMPLE_RATE
    native_rate = int(samplerate)
    need_resample = native_rate != target_rate

    # Calculate chunk size for ~80ms of audio at native rate
    native_chunk = int(native_rate * 0.08)

    detected = False
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        nonlocal detected
        if detected:
            return
        if status:
            print(status)

        # Convert to float and get mono
        audio = indata[:, 0]

        # Resample to 16kHz if needed
        if need_resample:
            num_samples = int(len(audio) * target_rate / native_rate)
            audio = signal.resample(audio, num_samples)

        # Convert to 16-bit int for model
        audio_data = (audio * 32767).astype(np.int16)

        # Feed to wake word model
        prediction = model.predict(audio_data)

        # Only trigger on the configured wake word
        if WAKE_WORD in prediction and prediction[WAKE_WORD] > WAKE_THRESHOLD:
            print(f"\n{wake(f'Detected: {WAKE_WORD} ({prediction[WAKE_WORD]:.2f})')}")
            detected = True
            return

    print(wake(f"Listening... (say '{WAKE_WORD.replace('_', ' ').title()}')"))

    try:
        with sd.InputStream(
            samplerate=native_rate,
            blocksize=native_chunk,
            device=selected_device['index'],
            channels=1,
            dtype='float32',
            callback=audio_callback
        ):
            while not detected:
                time.sleep(0.1)
                if timeout and (time.time() - start_time) > timeout:
                    return False

        return True

    except Exception as e:
        print(f"Wake word error: {e}")
        return None

def list_available_wakewords():
    """List available pre-trained wake words."""
    return [
        "hey_jarvis",
        "alexa",
        "hey_mycroft",
        "timer",
        "weather"
    ]
