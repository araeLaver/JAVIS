"""Conversation logger for fine-tuning data collection."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class ConversationTurn(BaseModel):
    """Single conversation turn."""
    role: str
    content: str
    timestamp: str


class ConversationLog(BaseModel):
    """Full conversation log."""
    session_id: str
    started_at: str
    turns: list[ConversationTurn] = []
    feedback: Optional[str] = None  # good, bad, None
    tags: list[str] = []


class ConversationLogger:
    """Logger for collecting fine-tuning data."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "conversations"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 현재 활성 대화들
        self.active_conversations: dict[str, ConversationLog] = {}

    def start_conversation(self, session_id: str) -> ConversationLog:
        """Start a new conversation."""
        conv = ConversationLog(
            session_id=session_id,
            started_at=datetime.now().isoformat()
        )
        self.active_conversations[session_id] = conv
        return conv

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a turn to the conversation."""
        if session_id not in self.active_conversations:
            self.start_conversation(session_id)

        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        self.active_conversations[session_id].turns.append(turn)

    def add_feedback(self, session_id: str, feedback: str):
        """Add feedback (good/bad) to a conversation."""
        if session_id in self.active_conversations:
            self.active_conversations[session_id].feedback = feedback

    def add_tags(self, session_id: str, tags: list[str]):
        """Add tags to a conversation."""
        if session_id in self.active_conversations:
            self.active_conversations[session_id].tags.extend(tags)

    def save_conversation(self, session_id: str):
        """Save conversation to disk."""
        if session_id not in self.active_conversations:
            return

        conv = self.active_conversations[session_id]

        # 날짜별 디렉토리
        date_dir = self.data_dir / datetime.now().strftime("%Y-%m")
        date_dir.mkdir(exist_ok=True)

        # 파일명: session_id_timestamp.json
        filename = f"{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = date_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conv.model_dump(), f, ensure_ascii=False, indent=2)

        return filepath

    def end_conversation(self, session_id: str):
        """End and save a conversation."""
        filepath = self.save_conversation(session_id)
        if session_id in self.active_conversations:
            del self.active_conversations[session_id]
        return filepath

    def export_for_training(self, output_path: Optional[Path] = None,
                           feedback_filter: Optional[str] = None) -> Path:
        """Export conversations in training format (JSONL)."""
        if output_path is None:
            output_path = self.data_dir.parent / "training" / "exported"
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"conversations_{datetime.now().strftime('%Y%m%d')}.jsonl"

        conversations = []

        # 모든 저장된 대화 읽기
        for json_file in self.data_dir.rglob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                conv = json.load(f)

                # 피드백 필터
                if feedback_filter and conv.get('feedback') != feedback_filter:
                    continue

                # 대화 턴이 2개 이상인 것만 (시스템 + 유저 + 어시스턴트)
                if len(conv.get('turns', [])) >= 2:
                    conversations.append(conv)

        # JSONL 형식으로 저장 (Qwen 학습 형식)
        with open(output_file, 'w', encoding='utf-8') as f:
            for conv in conversations:
                # Qwen 대화 형식으로 변환
                messages = []
                for turn in conv['turns']:
                    messages.append({
                        "role": turn['role'],
                        "content": turn['content']
                    })

                training_example = {"messages": messages}
                f.write(json.dumps(training_example, ensure_ascii=False) + '\n')

        return output_file


# 전역 로거 인스턴스
_logger: Optional[ConversationLogger] = None


def get_logger() -> ConversationLogger:
    """Get the global conversation logger."""
    global _logger
    if _logger is None:
        _logger = ConversationLogger()
    return _logger
