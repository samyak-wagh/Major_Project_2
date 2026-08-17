"""
Voice Handler
=============
Converts audio bytes (WAV/WebM from browser mic) to text
using Google's free Web Speech API via SpeechRecognition.
Requires internet for speech-to-text only (the OS-Tutor model stays offline).
"""

import io
import logging
import speech_recognition as sr

logger = logging.getLogger(__name__)


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert raw audio bytes to text.

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data captured from streamlit-mic-recorder (WAV format).

    Returns
    -------
    str
        Transcribed text, or empty string if nothing was recognised.

    Raises
    ------
    RuntimeError
        If the Google Speech API cannot be reached.
    """
    recognizer = sr.Recognizer()

    try:
        # Load audio from bytes
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        # Transcribe using Google's free API
        text = recognizer.recognize_google(audio_data, language="en-IN")
        logger.info(f"Transcribed: {text}")
        return text.strip()

    except sr.UnknownValueError:
        logger.warning("Speech not recognised — audio may be too quiet or unclear.")
        return ""
    except sr.RequestError as e:
        raise RuntimeError(
            f"Could not reach Google Speech API: {e}. Check your internet connection."
        )
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise RuntimeError(f"Audio processing failed: {e}")
