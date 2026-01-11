"""Preference data generation for DPO training."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from .corrections import CorrectionManager, get_correction_manager
from .feedback_store import FeedbackStore, get_feedback_store

logger = logging.getLogger(__name__)

PreferenceSource = Literal["correction", "feedback", "quality_score"]


class PreferencePair(BaseModel):
    """A preference pair for DPO training."""

    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Non-preferred response
    source: PreferenceSource
    metadata: dict = {}


class PreferenceStats(BaseModel):
    """Statistics about available preference data."""

    total_pairs: int = 0
    from_corrections: int = 0
    from_feedback: int = 0
    from_quality_score: int = 0
    ready_for_training: bool = False


class PreferenceDataGenerator:
    """Generates preference pairs from feedback and corrections."""

    def __init__(
        self,
        feedback_store: Optional[FeedbackStore] = None,
        correction_manager: Optional[CorrectionManager] = None,
        min_pairs: int = 50,
    ):
        self.feedback_store = feedback_store or get_feedback_store()
        self.correction_manager = correction_manager or get_correction_manager()
        self.min_pairs = min_pairs

    def from_corrections(self) -> list[PreferencePair]:
        """Generate preference pairs from corrections.

        Corrections provide explicit preference signals:
        - chosen = corrected_response
        - rejected = original_response

        Returns:
            List of PreferencePair objects
        """
        pairs = []
        corrections = self.correction_manager.get_corrections()

        for corr in corrections:
            # Skip if responses are identical
            if corr.original_response.strip() == corr.corrected_response.strip():
                continue

            pairs.append(
                PreferencePair(
                    prompt=corr.original_prompt,
                    chosen=corr.corrected_response,
                    rejected=corr.original_response,
                    source="correction",
                    metadata={
                        "correction_id": corr.id,
                        "correction_type": corr.correction_type,
                        "timestamp": corr.timestamp.isoformat(),
                    },
                )
            )

        logger.info(f"Generated {len(pairs)} preference pairs from corrections")
        return pairs

    def from_feedback_pairs(self) -> list[PreferencePair]:
        """Generate preference pairs from good/bad feedback.

        Matches prompts that have both good and bad responses.

        Returns:
            List of PreferencePair objects
        """
        pairs = []

        good_responses = self.feedback_store.get_good_responses()
        bad_responses = self.feedback_store.get_bad_responses()

        # Index good responses by prompt (normalized)
        good_by_prompt: dict[str, list] = {}
        for fb in good_responses:
            key = fb.prompt.strip().lower()
            if key not in good_by_prompt:
                good_by_prompt[key] = []
            good_by_prompt[key].append(fb)

        # Match bad responses with good responses for same prompt
        for bad_fb in bad_responses:
            key = bad_fb.prompt.strip().lower()
            if key in good_by_prompt:
                for good_fb in good_by_prompt[key]:
                    # Skip if responses are identical
                    if good_fb.response.strip() == bad_fb.response.strip():
                        continue

                    pairs.append(
                        PreferencePair(
                            prompt=good_fb.prompt,  # Use the original prompt from good
                            chosen=good_fb.response,
                            rejected=bad_fb.response,
                            source="feedback",
                            metadata={
                                "good_id": good_fb.id,
                                "bad_id": bad_fb.id,
                            },
                        )
                    )

        logger.info(f"Generated {len(pairs)} preference pairs from feedback matching")
        return pairs

    def from_corrected_feedback(self) -> list[PreferencePair]:
        """Generate preference pairs from feedback with corrections.

        Uses feedback entries that have corrected_response field.

        Returns:
            List of PreferencePair objects
        """
        pairs = []
        corrected = self.feedback_store.get_corrected_responses()

        for fb in corrected:
            if not fb.corrected_response:
                continue

            # Skip if responses are identical
            if fb.response.strip() == fb.corrected_response.strip():
                continue

            pairs.append(
                PreferencePair(
                    prompt=fb.prompt,
                    chosen=fb.corrected_response,
                    rejected=fb.response,
                    source="correction",
                    metadata={
                        "feedback_id": fb.id,
                        "timestamp": fb.timestamp.isoformat(),
                    },
                )
            )

        logger.info(f"Generated {len(pairs)} preference pairs from corrected feedback")
        return pairs

    def from_quality_scores(
        self, score_threshold: float = 3.0
    ) -> list[PreferencePair]:
        """Generate preference pairs from quality score differences.

        Pairs high-quality (score >= 4) with low-quality (score <= threshold)
        responses for similar prompts.

        Args:
            score_threshold: Maximum score to consider as "rejected"

        Returns:
            List of PreferencePair objects
        """
        pairs = []
        good_responses = self.feedback_store.get_good_responses()

        # Separate by quality score
        high_quality = [
            fb for fb in good_responses
            if fb.quality_score is not None and fb.quality_score >= 4.0
        ]
        low_quality = [
            fb for fb in good_responses
            if fb.quality_score is not None and fb.quality_score <= score_threshold
        ]

        # Index high quality by prompt
        high_by_prompt: dict[str, list] = {}
        for fb in high_quality:
            key = fb.prompt.strip().lower()
            if key not in high_by_prompt:
                high_by_prompt[key] = []
            high_by_prompt[key].append(fb)

        # Match low quality with high quality
        for low_fb in low_quality:
            key = low_fb.prompt.strip().lower()
            if key in high_by_prompt:
                for high_fb in high_by_prompt[key]:
                    if high_fb.response.strip() == low_fb.response.strip():
                        continue

                    pairs.append(
                        PreferencePair(
                            prompt=high_fb.prompt,
                            chosen=high_fb.response,
                            rejected=low_fb.response,
                            source="quality_score",
                            metadata={
                                "chosen_score": high_fb.quality_score,
                                "rejected_score": low_fb.quality_score,
                            },
                        )
                    )

        logger.info(f"Generated {len(pairs)} preference pairs from quality scores")
        return pairs

    def generate_all(self) -> list[PreferencePair]:
        """Generate all preference pairs from all sources.

        Returns:
            Combined list of PreferencePair objects (deduplicated)
        """
        all_pairs = []

        # Priority order: corrections > corrected feedback > feedback pairs > quality scores
        all_pairs.extend(self.from_corrections())
        all_pairs.extend(self.from_corrected_feedback())
        all_pairs.extend(self.from_feedback_pairs())
        all_pairs.extend(self.from_quality_scores())

        # Deduplicate by (prompt, chosen, rejected) tuple
        seen = set()
        unique_pairs = []
        for pair in all_pairs:
            key = (
                pair.prompt.strip().lower(),
                pair.chosen.strip(),
                pair.rejected.strip(),
            )
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        logger.info(
            f"Generated {len(unique_pairs)} unique preference pairs "
            f"(from {len(all_pairs)} total)"
        )
        return unique_pairs

    def get_statistics(self) -> PreferenceStats:
        """Get statistics about available preference data.

        Returns:
            PreferenceStats object
        """
        corrections = len(self.from_corrections())
        corrected_feedback = len(self.from_corrected_feedback())
        feedback_pairs = len(self.from_feedback_pairs())
        quality_pairs = len(self.from_quality_scores())

        total = corrections + corrected_feedback + feedback_pairs + quality_pairs

        return PreferenceStats(
            total_pairs=total,
            from_corrections=corrections + corrected_feedback,
            from_feedback=feedback_pairs,
            from_quality_score=quality_pairs,
            ready_for_training=total >= self.min_pairs,
        )

    def export_dpo_dataset(
        self,
        output_path: Optional[Path] = None,
        pairs: Optional[list[PreferencePair]] = None,
    ) -> Path:
        """Export preference pairs as DPO training JSONL.

        Args:
            output_path: Output file path
            pairs: Optional pre-generated pairs (will generate if not provided)

        Returns:
            Path to exported file
        """
        if output_path is None:
            output_dir = (
                Path(__file__).parent.parent.parent / "data" / "training" / "dpo"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (
                output_dir
                / f"preferences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )

        if pairs is None:
            pairs = self.generate_all()

        # Write in DPO format
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                data = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(pairs)} preference pairs to {output_path}")
        return output_path


# Singleton instance
_generator: Optional[PreferenceDataGenerator] = None


def get_preference_generator() -> PreferenceDataGenerator:
    """Get the global PreferenceDataGenerator instance."""
    global _generator
    if _generator is None:
        _generator = PreferenceDataGenerator()
    return _generator
