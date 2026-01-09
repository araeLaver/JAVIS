"""Edge TTS Provider.

Uses Microsoft Edge TTS for free, high-quality text-to-speech.
"""

import io
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional

from javis.utils.config import get_config
from javis.voice.base import TTSProvider

# Lazy import for optional dependency
edge_tts = None


def _ensure_edge_tts():
    """Ensure edge-tts dependency is available."""
    global edge_tts
    if edge_tts is None:
        try:
            import edge_tts as et

            edge_tts = et
        except ImportError:
            raise ImportError(
                "Edge TTS requires 'edge-tts'. Install with: pip install edge-tts"
            )


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS-based Text-to-Speech provider."""

    def __init__(
        self,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ):
        """Initialize Edge TTS provider.

        Args:
            voice: Voice name (e.g., "ko-KR-SunHiNeural")
            rate: Speech rate (e.g., "+0%", "+10%", "-10%")
            pitch: Voice pitch (e.g., "+0Hz", "+5Hz")
        """
        _ensure_edge_tts()

        config = get_config()
        edge_config = config.voice.tts.edge_tts

        self.voice = voice or edge_config.voice
        self.rate = rate or edge_config.rate
        self.pitch = pitch or edge_config.pitch

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes.

        Args:
            text: Text to convert to speech

        Returns:
            Audio data as MP3 bytes
        """
        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        audio_buffer = io.BytesIO()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])

        audio_buffer.seek(0)
        return audio_buffer.read()

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio in chunks.

        Args:
            text: Text to convert to speech

        Yields:
            Audio chunks as bytes
        """
        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def save_to_file(self, text: str, file_path: Path) -> Path:
        """Save synthesized audio to file.

        Args:
            text: Text to convert to speech
            file_path: Output file path

        Returns:
            Path to the saved audio file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
        )

        await communicate.save(str(file_path))
        return file_path

    async def save_to_temp_file(self, text: str) -> Path:
        """Save synthesized audio to a temporary file.

        Args:
            text: Text to convert to speech

        Returns:
            Path to the temporary audio file
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = Path(f.name)

        return await self.save_to_file(text, temp_path)

    @staticmethod
    async def list_voices(language: Optional[str] = None) -> list[dict]:
        """List available TTS voices.

        Args:
            language: Filter by language (e.g., "ko-KR", "en-US")

        Returns:
            List of voice info dictionaries
        """
        _ensure_edge_tts()

        voices = await edge_tts.list_voices()

        if language:
            voices = [v for v in voices if v.get("Locale", "").startswith(language)]

        return [
            {
                "name": v.get("ShortName"),
                "locale": v.get("Locale"),
                "gender": v.get("Gender"),
                "friendly_name": v.get("FriendlyName"),
            }
            for v in voices
        ]

    @staticmethod
    async def list_korean_voices() -> list[dict]:
        """List available Korean TTS voices.

        Returns:
            List of Korean voice info dictionaries
        """
        return await EdgeTTSProvider.list_voices("ko-KR")
