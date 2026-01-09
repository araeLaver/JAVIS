"""Voice session for CLI interactive voice mode.

Provides a continuous voice conversation experience.
"""

import asyncio
from typing import Callable, Optional

from javis.core.engine import ConversationEngine
from javis.utils.config import get_config
from javis.voice import get_stt_provider, get_tts_provider
from javis.voice.stt.recorder import AudioRecorder
from javis.voice.tts.player import AudioPlayer


class VoiceSession:
    """Interactive voice conversation session."""

    # Commands to exit voice mode (Korean)
    EXIT_COMMANDS = {"종료", "끝", "나가기", "exit", "quit", "stop"}

    def __init__(
        self,
        engine: Optional[ConversationEngine] = None,
        auto_play: bool = True,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
    ):
        """Initialize voice session.

        Args:
            engine: ConversationEngine instance (creates new if None)
            auto_play: Whether to automatically play TTS responses
            on_transcription: Callback when user speech is transcribed
            on_response: Callback when AI response is generated
        """
        config = get_config()

        if not config.voice.enabled:
            raise RuntimeError("Voice interface is not enabled in config")

        self.engine = engine or ConversationEngine()
        self.auto_play = auto_play

        # Initialize providers
        self.stt = get_stt_provider()
        self.tts = get_tts_provider()
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()

        # Callbacks
        self.on_transcription = on_transcription
        self.on_response = on_response

        # Session state
        self._running = False
        self._language = config.voice.language.split("-")[0]  # "ko-KR" -> "ko"

    async def start(self):
        """Start the voice conversation session."""
        self._running = True
        print("\n🎤 음성 대화 모드를 시작합니다.")
        print("   말씀하세요... ('종료' 또는 Ctrl+C로 종료)\n")

        # Initial greeting
        await self._speak("안녕하세요. 무엇을 도와드릴까요?")

        try:
            while self._running:
                await self._conversation_turn()
        except KeyboardInterrupt:
            print("\n\n👋 음성 대화를 종료합니다.")
        finally:
            self.stop()

    async def _conversation_turn(self):
        """Execute one turn of conversation."""
        try:
            # 1. Record user speech
            print("🎤 듣는 중... (말씀하세요)")
            audio_data = await self.recorder.record()

            if not audio_data:
                print("   ❌ 음성이 감지되지 않았습니다.\n")
                return

            # 2. Transcribe speech to text
            print("📝 인식 중...")
            result = await self.stt.transcribe(audio_data, self._language)
            user_text = result.text.strip()

            if not user_text:
                print("   ❌ 음성을 인식하지 못했습니다.\n")
                return

            print(f"   사용자: {user_text}\n")

            if self.on_transcription:
                self.on_transcription(user_text)

            # 3. Check for exit command
            if user_text.lower() in self.EXIT_COMMANDS:
                print("👋 음성 대화를 종료합니다.")
                self._running = False
                return

            # 4. Get AI response
            print("🤔 생각 중...")
            response = await self.engine.chat(user_text)
            print(f"   JAVIS: {response}\n")

            if self.on_response:
                self.on_response(response)

            # 5. Speak response
            if self.auto_play:
                await self._speak(response)

        except Exception as e:
            print(f"   ❌ 오류: {e}\n")

    async def _speak(self, text: str):
        """Synthesize and play text as speech.

        Args:
            text: Text to speak
        """
        try:
            print("🔊 말하는 중...")
            audio_data = await self.tts.synthesize(text)
            await self.player.play(audio_data)
        except Exception as e:
            print(f"   ⚠️ TTS 오류: {e}")

    async def speak(self, text: str):
        """Public method to speak text.

        Args:
            text: Text to speak
        """
        await self._speak(text)

    async def listen(self) -> Optional[str]:
        """Listen and transcribe user speech.

        Returns:
            Transcribed text or None if nothing detected
        """
        audio_data = await self.recorder.record()
        if not audio_data:
            return None

        result = await self.stt.transcribe(audio_data, self._language)
        return result.text.strip() if result.text else None

    def stop(self):
        """Stop the voice session."""
        self._running = False
        self.recorder.stop()
        self.player.stop()

    @property
    def is_running(self) -> bool:
        """Check if session is running."""
        return self._running


async def run_voice_mode(tts_voice: Optional[str] = None):
    """Run voice mode from CLI.

    Args:
        tts_voice: Optional TTS voice override
    """
    from javis.core.engine import ConversationEngine

    # Override TTS voice if specified
    if tts_voice:
        from javis.voice.tts.edge_tts_provider import EdgeTTSProvider

        # Create custom TTS provider with specified voice
        custom_tts = EdgeTTSProvider(voice=tts_voice)

        # Temporarily replace the global provider
        import javis.voice as voice_module
        voice_module._tts_provider = custom_tts

    engine = ConversationEngine()
    session = VoiceSession(engine=engine)
    await session.start()
