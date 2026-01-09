"""Groq Whisper STT Provider.

Uses Groq's Whisper API for fast, accurate speech-to-text conversion.
"""

import io
from pathlib import Path
from typing import Optional

import httpx

from javis.utils.config import get_config
from javis.voice.base import STTProvider, TranscriptionResult


class GroqWhisperSTT(STTProvider):
    """Groq Whisper API-based Speech-to-Text provider."""

    API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self):
        """Initialize Groq Whisper STT provider."""
        config = get_config()
        self.api_key = config.groq_api_key
        self.model = config.voice.stt.groq_whisper.model
        self.default_language = config.voice.language.split("-")[0]  # "ko-KR" -> "ko"

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for Groq Whisper STT")

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "ko",
        filename: str = "audio.wav",
    ) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes (WAV format recommended)
            language: Language code (e.g., "ko", "en")
            filename: Filename hint for the API

        Returns:
            TranscriptionResult with transcribed text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Prepare multipart form data
        files = {
            "file": (filename, io.BytesIO(audio_data), "audio/wav"),
        }
        data = {
            "model": self.model,
            "language": language or self.default_language,
            "response_format": "verbose_json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.API_URL,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            result = response.json()

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language"),
            duration=result.get("duration"),
        )

    async def transcribe_file(
        self,
        file_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            file_path: Path to audio file
            language: Language code (optional, uses default if not specified)

        Returns:
            TranscriptionResult with transcribed text
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        with open(file_path, "rb") as f:
            audio_data = f.read()

        return await self.transcribe(
            audio_data=audio_data,
            language=language or self.default_language,
            filename=file_path.name,
        )
