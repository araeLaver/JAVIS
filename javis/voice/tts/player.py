"""Audio player utility for TTS.

Provides audio playback functionality using pygame.
"""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

# Lazy imports for optional dependencies
pygame = None
pygame_mixer = None


def _ensure_pygame():
    """Ensure pygame dependency is available."""
    global pygame, pygame_mixer
    if pygame is None:
        try:
            import pygame as pg

            pg.mixer.init()
            pygame = pg
            pygame_mixer = pg.mixer
        except ImportError:
            raise ImportError(
                "Audio playback requires 'pygame'. Install with: pip install pygame"
            )


class AudioPlayer:
    """Audio player for TTS output using pygame."""

    def __init__(self):
        """Initialize audio player."""
        _ensure_pygame()
        self._current_sound: Optional[object] = None
        self._is_playing = False

    async def play(self, audio_data: bytes, format: str = "mp3"):
        """Play audio from bytes.

        Args:
            audio_data: Audio data bytes
            format: Audio format (mp3, wav, ogg)
        """
        # Save to temp file since pygame works better with files
        with tempfile.NamedTemporaryFile(
            suffix=f".{format}", delete=False
        ) as f:
            f.write(audio_data)
            temp_path = Path(f.name)

        try:
            await self.play_file(temp_path)
        finally:
            # Clean up temp file
            try:
                temp_path.unlink()
            except Exception:
                pass

    async def play_file(self, file_path: Path):
        """Play audio from file.

        Args:
            file_path: Path to audio file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        def _play():
            pygame_mixer.music.load(str(file_path))
            pygame_mixer.music.play()

        # Load and play in executor
        await asyncio.get_event_loop().run_in_executor(None, _play)
        self._is_playing = True

        # Wait for playback to finish
        while pygame_mixer.music.get_busy():
            await asyncio.sleep(0.1)

        self._is_playing = False

    async def play_stream(self, audio_stream, chunk_callback=None):
        """Play streaming audio.

        Note: For simplicity, this buffers the stream then plays.
        For true streaming, a more complex approach is needed.

        Args:
            audio_stream: Async generator yielding audio chunks
            chunk_callback: Optional callback for each chunk
        """
        # Buffer all chunks
        audio_buffer = io.BytesIO()

        async for chunk in audio_stream:
            audio_buffer.write(chunk)
            if chunk_callback:
                chunk_callback(len(chunk))

        # Play the complete audio
        audio_buffer.seek(0)
        await self.play(audio_buffer.read())

    def stop(self):
        """Stop current playback."""
        if self._is_playing:
            pygame_mixer.music.stop()
            self._is_playing = False

    def pause(self):
        """Pause current playback."""
        if self._is_playing:
            pygame_mixer.music.pause()

    def resume(self):
        """Resume paused playback."""
        pygame_mixer.music.unpause()

    def set_volume(self, volume: float):
        """Set playback volume.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        pygame_mixer.music.set_volume(max(0.0, min(1.0, volume)))

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._is_playing and pygame_mixer.music.get_busy()

    @staticmethod
    def cleanup():
        """Clean up pygame mixer resources."""
        if pygame_mixer is not None:
            pygame_mixer.quit()
