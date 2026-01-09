"""Audio recorder utility for STT.

Provides microphone recording functionality with silence detection.
"""

import asyncio
import io
import queue
import threading
from typing import Optional

from javis.utils.config import get_config

# Lazy imports for optional dependencies
sounddevice = None
soundfile = None


def _ensure_audio_deps():
    """Ensure audio dependencies are available."""
    global sounddevice, soundfile
    if sounddevice is None:
        try:
            import sounddevice as sd
            import soundfile as sf

            sounddevice = sd
            soundfile = sf
        except ImportError:
            raise ImportError(
                "Audio recording requires 'sounddevice' and 'soundfile'. "
                "Install with: pip install sounddevice soundfile"
            )


class AudioRecorder:
    """Microphone audio recorder with silence detection."""

    def __init__(
        self,
        sample_rate: Optional[int] = None,
        silence_threshold: Optional[int] = None,
        max_duration: Optional[int] = None,
    ):
        """Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz (default from config)
            silence_threshold: Silence detection threshold in ms (default from config)
            max_duration: Maximum recording duration in seconds (default from config)
        """
        _ensure_audio_deps()

        config = get_config()
        recording_config = config.voice.stt.recording

        self.sample_rate = sample_rate or recording_config.sample_rate
        self.silence_threshold_ms = silence_threshold or recording_config.silence_threshold
        self.max_duration = max_duration or recording_config.max_duration

        self._audio_queue: queue.Queue = queue.Queue()
        self._recording = False
        self._stop_event = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        self._audio_queue.put(indata.copy())

    def _calculate_rms(self, audio_data) -> float:
        """Calculate RMS (root mean square) of audio data."""
        import numpy as np

        return float(np.sqrt(np.mean(audio_data**2)))

    async def record(self, silence_duration_ms: Optional[int] = None) -> bytes:
        """Record audio from microphone until silence is detected.

        Args:
            silence_duration_ms: Duration of silence to stop recording (default: silence_threshold)

        Returns:
            Recorded audio as WAV bytes
        """
        import numpy as np

        silence_duration = (silence_duration_ms or self.silence_threshold_ms) / 1000.0
        silence_threshold_rms = 0.01  # RMS threshold for silence

        self._recording = True
        self._stop_event.clear()
        self._audio_queue = queue.Queue()

        audio_chunks = []
        silent_chunks = 0
        chunks_per_second = 10
        chunk_duration = 1.0 / chunks_per_second
        samples_per_chunk = int(self.sample_rate * chunk_duration)
        silence_chunks_needed = int(silence_duration / chunk_duration)
        max_chunks = int(self.max_duration / chunk_duration)

        # Start recording in a thread
        stream = sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=samples_per_chunk,
            callback=self._audio_callback,
        )

        try:
            stream.start()
            total_chunks = 0

            while self._recording and total_chunks < max_chunks:
                try:
                    # Get audio chunk with timeout
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._audio_queue.get(timeout=0.5)
                    )
                    audio_chunks.append(chunk)
                    total_chunks += 1

                    # Check for silence
                    rms = self._calculate_rms(chunk)
                    if rms < silence_threshold_rms:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0

                    # Stop if enough silence detected (after some audio was recorded)
                    if silent_chunks >= silence_chunks_needed and len(audio_chunks) > chunks_per_second:
                        break

                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    continue

        finally:
            stream.stop()
            stream.close()
            self._recording = False

        if not audio_chunks:
            return b""

        # Concatenate and convert to WAV
        audio_data = np.concatenate(audio_chunks, axis=0)
        return self._to_wav_bytes(audio_data)

    async def record_for_duration(self, duration: float) -> bytes:
        """Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            Recorded audio as WAV bytes
        """
        import numpy as np

        samples = int(duration * self.sample_rate)

        # Record synchronously in executor
        def _record():
            return sounddevice.rec(
                samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )

        audio_data = await asyncio.get_event_loop().run_in_executor(None, _record)

        # Wait for recording to complete
        await asyncio.get_event_loop().run_in_executor(None, sounddevice.wait)

        return self._to_wav_bytes(audio_data)

    def _to_wav_bytes(self, audio_data) -> bytes:
        """Convert numpy audio data to WAV bytes.

        Args:
            audio_data: Numpy array of audio samples

        Returns:
            WAV file as bytes
        """
        buffer = io.BytesIO()
        soundfile.write(buffer, audio_data, self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def stop(self):
        """Stop recording."""
        self._recording = False
        self._stop_event.set()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices.

        Returns:
            List of device info dictionaries
        """
        _ensure_audio_deps()

        devices = sounddevice.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )

        return input_devices
