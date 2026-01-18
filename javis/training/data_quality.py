"""Data quality scoring system for training data."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from javis.storage import get_conversation_store, get_quality_score_store
from javis.storage.base import (
    ConversationStoreInterface,
    QualityScoreRecord,
    QualityScoreStoreInterface,
)

logger = logging.getLogger(__name__)


class DataQualityScore(BaseModel):
    """Quality score for a conversation."""

    conversation_id: str
    overall_score: float  # 1.0-5.0
    length_score: float  # Based on turn count and content length
    coherence_score: float  # Logical flow of conversation
    feedback_score: float  # Based on user feedback
    recency_score: float  # More recent = higher score
    computed_at: datetime


class DataQualityConfig(BaseModel):
    """Configuration for quality scoring."""

    min_turns: int = 2
    max_turns: int = 50
    ideal_turn_length: int = 200  # characters
    min_turn_length: int = 10
    max_turn_length: int = 2000
    feedback_weight: float = 0.4
    length_weight: float = 0.2
    coherence_weight: float = 0.2
    recency_weight: float = 0.2
    recency_days: int = 90  # Days for recency calculation


class DataQualityScorer:
    """Scores training data quality.

    Uses the storage factory to read conversations from either local files
    or cloud storage (Supabase) depending on configuration.
    """

    def __init__(
        self,
        config: Optional[DataQualityConfig] = None,
        conversations_dir: Optional[Path] = None,
        conversation_store: Optional[ConversationStoreInterface] = None,
        quality_store: Optional[QualityScoreStoreInterface] = None,
    ):
        self.config = config or DataQualityConfig()

        # Store references (lazy initialization)
        self._conversation_store = conversation_store
        self._quality_store = quality_store

        # Keep conversations_dir for backward compatibility
        if conversations_dir is None:
            conversations_dir = (
                Path(__file__).parent.parent.parent / "data" / "conversations"
            )
        self.conversations_dir = Path(conversations_dir)

    @property
    def conversation_store(self) -> ConversationStoreInterface:
        """Get the conversation store (lazy initialization)."""
        if self._conversation_store is None:
            self._conversation_store = get_conversation_store()
        return self._conversation_store

    @property
    def quality_store(self) -> QualityScoreStoreInterface:
        """Get the quality score store (lazy initialization)."""
        if self._quality_store is None:
            self._quality_store = get_quality_score_store()
        return self._quality_store

    def _calculate_length_score(self, conversation: dict) -> float:
        """Calculate score based on conversation length.

        Args:
            conversation: Conversation data

        Returns:
            Score from 1.0 to 5.0
        """
        turns = conversation.get("turns", [])
        turn_count = len(turns)

        # Score based on turn count
        if turn_count < self.config.min_turns:
            return 1.0
        elif turn_count > self.config.max_turns:
            turn_score = 3.0  # Too long is not ideal
        else:
            # Ideal range: 4-10 turns
            if 4 <= turn_count <= 10:
                turn_score = 5.0
            elif turn_count < 4:
                turn_score = 3.0 + (turn_count - self.config.min_turns)
            else:
                turn_score = max(3.0, 5.0 - (turn_count - 10) * 0.1)

        # Score based on content length
        total_length = sum(len(turn.get("content", "")) for turn in turns)
        avg_length = total_length / turn_count if turn_count > 0 else 0

        if avg_length < self.config.min_turn_length:
            length_score = 1.0
        elif avg_length > self.config.max_turn_length:
            length_score = 3.0
        elif abs(avg_length - self.config.ideal_turn_length) < 50:
            length_score = 5.0
        else:
            # Calculate based on distance from ideal
            distance = abs(avg_length - self.config.ideal_turn_length)
            length_score = max(2.0, 5.0 - (distance / 100))

        return (turn_score + length_score) / 2

    def _calculate_coherence_score(self, conversation: dict) -> float:
        """Calculate score based on conversation coherence.

        Args:
            conversation: Conversation data

        Returns:
            Score from 1.0 to 5.0
        """
        turns = conversation.get("turns", [])

        if len(turns) < 2:
            return 1.0

        score = 5.0

        # Check for alternating roles
        prev_role = None
        for turn in turns:
            role = turn.get("role")
            if role == prev_role:
                score -= 0.5  # Penalty for same role twice
            prev_role = role

        # Check for very short responses (might indicate issues)
        for turn in turns:
            content = turn.get("content", "")
            if len(content) < 5:
                score -= 0.3

        # Check for assistant responses to user questions
        user_questions = 0
        assistant_responses = 0
        for i, turn in enumerate(turns):
            if turn.get("role") == "user" and "?" in turn.get("content", ""):
                user_questions += 1
                # Check if next turn is assistant response
                if i + 1 < len(turns) and turns[i + 1].get("role") == "assistant":
                    assistant_responses += 1

        if user_questions > 0:
            response_rate = assistant_responses / user_questions
            if response_rate < 0.8:
                score -= 1.0

        return max(1.0, min(5.0, score))

    def _calculate_feedback_score(self, conversation: dict) -> float:
        """Calculate score based on user feedback.

        Args:
            conversation: Conversation data

        Returns:
            Score from 1.0 to 5.0
        """
        feedback = conversation.get("feedback")

        if feedback == "good":
            return 5.0
        elif feedback == "bad":
            return 1.0
        else:
            # No feedback - neutral score
            return 3.0

    def _calculate_recency_score(self, conversation: dict) -> float:
        """Calculate score based on conversation recency.

        Args:
            conversation: Conversation data

        Returns:
            Score from 1.0 to 5.0
        """
        started_at = conversation.get("started_at")
        if not started_at:
            return 3.0  # Default if no timestamp

        try:
            conv_date = datetime.fromisoformat(started_at)
            days_old = (datetime.now() - conv_date).days

            if days_old < 7:
                return 5.0
            elif days_old < 30:
                return 4.5
            elif days_old < 60:
                return 4.0
            elif days_old < self.config.recency_days:
                return 3.0
            else:
                return 2.0
        except (ValueError, TypeError):
            return 3.0

    def score_conversation(self, conversation: dict) -> DataQualityScore:
        """Score a single conversation.

        Args:
            conversation: Conversation data dictionary

        Returns:
            DataQualityScore with all component scores
        """
        conversation_id = conversation.get("session_id", "unknown")

        length_score = self._calculate_length_score(conversation)
        coherence_score = self._calculate_coherence_score(conversation)
        feedback_score = self._calculate_feedback_score(conversation)
        recency_score = self._calculate_recency_score(conversation)

        # Weighted average
        overall_score = (
            length_score * self.config.length_weight
            + coherence_score * self.config.coherence_weight
            + feedback_score * self.config.feedback_weight
            + recency_score * self.config.recency_weight
        )

        return DataQualityScore(
            conversation_id=conversation_id,
            overall_score=round(overall_score, 2),
            length_score=round(length_score, 2),
            coherence_score=round(coherence_score, 2),
            feedback_score=round(feedback_score, 2),
            recency_score=round(recency_score, 2),
            computed_at=datetime.now(),
        )

    def score_all_conversations(self, use_cache: bool = True) -> list[DataQualityScore]:
        """Score all conversations using the storage backend.

        Args:
            use_cache: Whether to use cached scores when available

        Returns:
            List of DataQualityScore for all conversations
        """
        scores = []

        # Get all conversations from storage
        conversations = self.conversation_store.list_all()

        for conv in conversations:
            try:
                # Check cache if enabled
                if use_cache:
                    cached = self.quality_store.get(conv.session_id)
                    if cached:
                        scores.append(
                            DataQualityScore(
                                conversation_id=cached.conversation_id,
                                overall_score=cached.overall_score,
                                length_score=cached.length_score,
                                coherence_score=cached.coherence_score,
                                feedback_score=cached.feedback_score,
                                recency_score=cached.recency_score,
                                computed_at=cached.computed_at,
                            )
                        )
                        continue

                # Convert to dict for scoring
                conv_dict = {
                    "session_id": conv.session_id,
                    "started_at": conv.started_at,
                    "turns": [
                        {"role": t.role, "content": t.content, "timestamp": t.timestamp}
                        for t in conv.turns
                    ],
                    "feedback": conv.feedback,
                    "tags": conv.tags,
                }

                score = self.score_conversation(conv_dict)
                scores.append(score)

                # Cache the score
                self.quality_store.save(
                    QualityScoreRecord(
                        conversation_id=score.conversation_id,
                        overall_score=score.overall_score,
                        length_score=score.length_score,
                        coherence_score=score.coherence_score,
                        feedback_score=score.feedback_score,
                        recency_score=score.recency_score,
                        computed_at=score.computed_at,
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to score conversation {conv.session_id}: {e}")
                continue

        logger.info(f"Scored {len(scores)} conversations")
        return scores

    def get_high_quality_ids(self, min_score: float = 3.5) -> list[str]:
        """Get IDs of high-quality conversations.

        Args:
            min_score: Minimum overall score threshold

        Returns:
            List of conversation IDs meeting the threshold
        """
        # Try to use storage backend's optimized method first
        try:
            return self.quality_store.get_high_quality_ids(min_score)
        except Exception:
            # Fallback to scoring all conversations
            scores = self.score_all_conversations()
            return [s.conversation_id for s in scores if s.overall_score >= min_score]

    def get_statistics(self) -> dict:
        """Get statistics about data quality.

        Returns:
            Dictionary with quality statistics
        """
        scores = self.score_all_conversations()

        if not scores:
            return {
                "total": 0,
                "high_quality": 0,
                "medium_quality": 0,
                "low_quality": 0,
                "average_score": 0,
            }

        overall_scores = [s.overall_score for s in scores]

        return {
            "total": len(scores),
            "high_quality": sum(1 for s in overall_scores if s >= 4.0),
            "medium_quality": sum(1 for s in overall_scores if 3.0 <= s < 4.0),
            "low_quality": sum(1 for s in overall_scores if s < 3.0),
            "average_score": round(sum(overall_scores) / len(overall_scores), 2),
            "min_score": round(min(overall_scores), 2),
            "max_score": round(max(overall_scores), 2),
        }

    def export_quality_report(self, output_path: Optional[Path] = None) -> Path:
        """Export quality scores to a JSON file.

        Args:
            output_path: Output file path

        Returns:
            Path to the exported file
        """
        if output_path is None:
            output_dir = (
                Path(__file__).parent.parent.parent / "data" / "quality_scores"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        scores = self.score_all_conversations()
        stats = self.get_statistics()

        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": stats,
            "scores": [s.model_dump() for s in scores],
        }

        # Convert datetime to string for JSON serialization
        for score in report["scores"]:
            score["computed_at"] = score["computed_at"].isoformat()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported quality report to {output_path}")
        return output_path


# Singleton instance
_scorer: Optional[DataQualityScorer] = None


def get_quality_scorer() -> DataQualityScorer:
    """Get the global DataQualityScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = DataQualityScorer()
    return _scorer
