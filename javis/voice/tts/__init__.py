"""Text-to-Speech module."""

from javis.voice import get_tts_provider
from javis.voice.tts.edge_tts_provider import EdgeTTSProvider
from javis.voice.tts.player import AudioPlayer

__all__ = ["get_tts_provider", "EdgeTTSProvider", "AudioPlayer"]
