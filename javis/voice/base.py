"""Base classes for voice interface providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Optional

from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    """Result from speech-to-text transcription."""

    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "ko",
    ) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes
            language: Language code (default: Korean)

        Returns:
            TranscriptionResult with transcribed text
        """
        pass

    @abstractmethod
    async def transcribe_file(
        self,
        file_path: Path,
        language: str = "ko",
    ) -> TranscriptionResult:
        """Transcribe audio file to text.

        Args:
            file_path: Path to audio file
            language: Language code (default: Korean)

        Returns:
            TranscriptionResult with transcribed text
        """
        pass


class TTSProvider(ABC):
    """Abstract base class for Text-to-Speech providers."""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes
        """
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to streaming audio.

        Args:
            text: Text to synthesize

        Yields:
            Audio data chunks
        """
        pass

    @abstractmethod
    async def save_to_file(
        self,
        text: str,
        file_path: Path,
    ) -> Path:
        """Synthesize text and save to file.

        Args:
            text: Text to synthesize
            file_path: Output file path

        Returns:
            Path to saved audio file
        """
        pass
