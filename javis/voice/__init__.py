"""Voice interface module for JAVIS.

Provides Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities.

Usage:
    # STT (Speech-to-Text)
    from javis.voice import get_stt_provider
    stt = get_stt_provider()
    result = await stt.transcribe_file("audio.wav")
    print(result.text)

    # TTS (Text-to-Speech)
    from javis.voice import get_tts_provider
    tts = get_tts_provider()
    await tts.save_to_file("안녕하세요", "output.mp3")

    # Voice Session (CLI)
    from javis.voice.session import VoiceSession
    session = VoiceSession()
    await session.start()
"""

from typing import Optional

from javis.utils.config import get_config

from .base import STTProvider, TTSProvider, TranscriptionResult

__all__ = [
    "STTProvider",
    "TTSProvider",
    "TranscriptionResult",
    "get_stt_provider",
    "get_tts_provider",
]

# Singleton instances
_stt_provider: Optional[STTProvider] = None
_tts_provider: Optional[TTSProvider] = None


def get_stt_provider() -> STTProvider:
    """Get the configured STT provider instance.

    Returns:
        STTProvider instance based on configuration
    """
    global _stt_provider

    if _stt_provider is None:
        config = get_config()
        provider = config.voice.stt.provider

        if provider == "groq_whisper":
            from .stt.groq_whisper import GroqWhisperSTT

            _stt_provider = GroqWhisperSTT()
        else:
            raise ValueError(f"Unknown STT provider: {provider}")

    return _stt_provider


def get_tts_provider() -> TTSProvider:
    """Get the configured TTS provider instance.

    Returns:
        TTSProvider instance based on configuration
    """
    global _tts_provider

    if _tts_provider is None:
        config = get_config()
        provider = config.voice.tts.provider

        if provider == "edge_tts":
            from .tts.edge_tts_provider import EdgeTTSProvider

            _tts_provider = EdgeTTSProvider()
        else:
            raise ValueError(f"Unknown TTS provider: {provider}")

    return _tts_provider
