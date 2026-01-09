"""Model validation system for testing trained models before deployment."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from javis.utils.config import get_config

logger = logging.getLogger(__name__)


class ValidationPrompt(BaseModel):
    """Test prompt for model validation."""

    prompt: str
    expected_keywords: list[str] = []  # Words that should appear
    forbidden_keywords: list[str] = []  # Words that should NOT appear
    max_latency_ms: int = 10000
    category: str = "general"  # general, code, korean, etc.


class ValidationTestResult(BaseModel):
    """Result of a single validation test."""

    prompt: str
    response: str
    passed: bool
    latency_ms: float
    category: str
    failure_reason: Optional[str] = None


class ValidationResult(BaseModel):
    """Result of model validation."""

    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: list[ValidationTestResult]
    average_latency_ms: float
    validation_time: datetime
    version: str


class ModelValidator:
    """Validates trained models before deployment."""

    def __init__(self, prompts_path: Optional[Path] = None):
        config = get_config()
        self.config = config.training.validation

        if prompts_path is None:
            prompts_path = Path(self.config.prompts_path)
            if not prompts_path.is_absolute():
                prompts_path = Path(__file__).parent.parent.parent / prompts_path

        self.prompts_path = prompts_path
        self._prompts: Optional[list[ValidationPrompt]] = None

    def _get_default_prompts(self) -> list[ValidationPrompt]:
        """Get default validation prompts."""
        return [
            ValidationPrompt(
                prompt="What is 2+2?",
                expected_keywords=["4"],
                max_latency_ms=5000,
                category="general",
            ),
            ValidationPrompt(
                prompt="Write a Python function to add two numbers.",
                expected_keywords=["def", "return"],
                max_latency_ms=10000,
                category="code",
            ),
            ValidationPrompt(
                prompt="Say hello in Korean.",
                expected_keywords=["안녕"],
                max_latency_ms=5000,
                category="korean",
            ),
            ValidationPrompt(
                prompt="What is the capital of South Korea?",
                expected_keywords=["Seoul", "서울"],
                max_latency_ms=5000,
                category="general",
            ),
            ValidationPrompt(
                prompt="Explain what a for loop does in programming.",
                expected_keywords=["loop", "iterate", "repeat"],
                max_latency_ms=8000,
                category="code",
            ),
        ]

    def load_validation_prompts(self) -> list[ValidationPrompt]:
        """Load validation prompts from file or use defaults.

        Returns:
            List of ValidationPrompt objects
        """
        if self._prompts is not None:
            return self._prompts

        if self.prompts_path.exists():
            try:
                with open(self.prompts_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._prompts = [ValidationPrompt(**p) for p in data]
                logger.info(f"Loaded {len(self._prompts)} validation prompts from {self.prompts_path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load prompts from {self.prompts_path}: {e}")
                self._prompts = self._get_default_prompts()
        else:
            logger.info("Using default validation prompts")
            self._prompts = self._get_default_prompts()

        return self._prompts

    def check_output_quality(
        self, response: str, expected: ValidationPrompt
    ) -> tuple[bool, Optional[str]]:
        """Check if response meets quality criteria.

        Args:
            response: Model response
            expected: ValidationPrompt with criteria

        Returns:
            (passed, failure_reason) tuple
        """
        response_lower = response.lower()

        # Check expected keywords (any of them should appear)
        if expected.expected_keywords:
            found = any(
                kw.lower() in response_lower for kw in expected.expected_keywords
            )
            if not found:
                return False, f"Missing expected keywords: {expected.expected_keywords}"

        # Check forbidden keywords (none should appear)
        for kw in expected.forbidden_keywords:
            if kw.lower() in response_lower:
                return False, f"Found forbidden keyword: {kw}"

        # Check response is not empty or too short
        if len(response.strip()) < 5:
            return False, "Response too short"

        return True, None

    def validate_with_inference(
        self,
        inference_fn,
        version: str,
    ) -> ValidationResult:
        """Validate model using an inference function.

        Args:
            inference_fn: Function that takes a prompt and returns a response
            version: Version being validated

        Returns:
            ValidationResult with test outcomes
        """
        prompts = self.load_validation_prompts()
        results = []
        total_latency = 0

        for prompt_config in prompts:
            start_time = time.time()

            try:
                response = inference_fn(prompt_config.prompt)
                latency_ms = (time.time() - start_time) * 1000

                # Check quality
                passed, failure_reason = self.check_output_quality(response, prompt_config)

                # Check latency
                if latency_ms > prompt_config.max_latency_ms:
                    passed = False
                    failure_reason = f"Latency {latency_ms:.0f}ms exceeds max {prompt_config.max_latency_ms}ms"

                result = ValidationTestResult(
                    prompt=prompt_config.prompt,
                    response=response,
                    passed=passed,
                    latency_ms=latency_ms,
                    category=prompt_config.category,
                    failure_reason=failure_reason,
                )

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                result = ValidationTestResult(
                    prompt=prompt_config.prompt,
                    response="",
                    passed=False,
                    latency_ms=latency_ms,
                    category=prompt_config.category,
                    failure_reason=f"Inference error: {str(e)}",
                )

            results.append(result)
            total_latency += result.latency_ms

        passed_count = sum(1 for r in results if r.passed)
        failed_tests = [r for r in results if not r.passed]

        # Determine overall pass/fail
        pass_rate = passed_count / len(results) if results else 0
        overall_passed = (
            pass_rate >= self.config.required_pass_rate
            and len(failed_tests) <= self.config.max_failures
        )

        return ValidationResult(
            passed=overall_passed,
            total_tests=len(results),
            passed_tests=passed_count,
            failed_tests=failed_tests,
            average_latency_ms=total_latency / len(results) if results else 0,
            validation_time=datetime.now(),
            version=version,
        )

    def validate_files_only(self, adapter_path: Path, version: str) -> ValidationResult:
        """Validate that required model files exist (fallback validation).

        Args:
            adapter_path: Path to adapter directory
            version: Version being validated

        Returns:
            ValidationResult based on file checks
        """
        required_files = ["adapter_config.json"]
        optional_files = ["adapter_model.safetensors", "adapter_model.bin"]

        failed_tests = []

        for req_file in required_files:
            file_path = adapter_path / req_file
            passed = file_path.exists()

            if not passed:
                failed_tests.append(
                    ValidationTestResult(
                        prompt=f"Check file: {req_file}",
                        response="",
                        passed=False,
                        latency_ms=0,
                        category="file_check",
                        failure_reason=f"Required file missing: {req_file}",
                    )
                )

        # Check that at least one model file exists
        has_model_file = any(
            (adapter_path / f).exists() for f in optional_files
        )

        if not has_model_file:
            failed_tests.append(
                ValidationTestResult(
                    prompt="Check model files",
                    response="",
                    passed=False,
                    latency_ms=0,
                    category="file_check",
                    failure_reason=f"No model file found (expected one of: {optional_files})",
                )
            )

        total_checks = len(required_files) + 1  # +1 for model file check
        passed_count = total_checks - len(failed_tests)

        return ValidationResult(
            passed=len(failed_tests) == 0,
            total_tests=total_checks,
            passed_tests=passed_count,
            failed_tests=failed_tests,
            average_latency_ms=0,
            validation_time=datetime.now(),
            version=version,
        )

    def save_validation_result(
        self, result: ValidationResult, output_dir: Optional[Path] = None
    ) -> Path:
        """Save validation result to file.

        Args:
            result: ValidationResult to save
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"validation_{result.version}_{result.validation_time.strftime('%Y%m%d_%H%M%S')}.json"
        file_path = output_dir / filename

        data = result.model_dump()
        data["validation_time"] = data["validation_time"].isoformat()
        for test in data["failed_tests"]:
            # Ensure all fields are serializable
            pass

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved validation result to {file_path}")
        return file_path


def save_default_prompts(output_path: Optional[Path] = None) -> Path:
    """Save default validation prompts to a JSON file.

    Args:
        output_path: Output file path

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / "configs" / "validation_prompts.json"

    validator = ModelValidator()
    prompts = validator._get_default_prompts()

    data = [p.model_dump() for p in prompts]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved default prompts to {output_path}")
    return output_path


# Singleton instance
_validator: Optional[ModelValidator] = None


def get_validator() -> ModelValidator:
    """Get the global ModelValidator instance."""
    global _validator
    if _validator is None:
        _validator = ModelValidator()
    return _validator
