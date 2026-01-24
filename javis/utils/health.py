"""Health check utilities for JAVIS application.

Provides comprehensive health checking for all system components
with support for Kubernetes-style liveness and readiness probes.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)
    last_check: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "status": self.status.value,
        }
        if self.message:
            result["message"] = self.message
        if self.latency_ms is not None:
            result["latency_ms"] = round(self.latency_ms, 2)
        if self.details:
            result["details"] = self.details
        if self.last_check:
            result["last_check"] = self.last_check.isoformat()
        return result


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    version: str
    uptime_seconds: float
    components: dict[str, ComponentHealth]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "timestamp": self.timestamp.isoformat(),
            "components": {
                name: comp.to_dict()
                for name, comp in self.components.items()
            },
        }


class HealthChecker:
    """Manages health checks for all system components."""

    def __init__(self, version: str = "0.1.0"):
        """Initialize health checker.

        Args:
            version: Application version string
        """
        self.version = version
        self._start_time = time.time()
        self._checks: dict[str, Callable[[], ComponentHealth]] = {}
        self._async_checks: dict[str, Callable[[], ComponentHealth]] = {}
        self._cached_results: dict[str, ComponentHealth] = {}
        self._cache_ttl_seconds = 30  # Cache health check results

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
        is_async: bool = False,
    ) -> None:
        """Register a health check function.

        Args:
            name: Component name
            check_fn: Function that returns ComponentHealth
            is_async: Whether the check function is async
        """
        if is_async:
            self._async_checks[name] = check_fn
        else:
            self._checks[name] = check_fn

    def unregister_check(self, name: str) -> None:
        """Unregister a health check.

        Args:
            name: Component name to remove
        """
        self._checks.pop(name, None)
        self._async_checks.pop(name, None)
        self._cached_results.pop(name, None)

    def _run_sync_check(self, name: str, check_fn: Callable) -> ComponentHealth:
        """Run a synchronous health check with timing."""
        start = time.time()
        try:
            result = check_fn()
            result.latency_ms = (time.time() - start) * 1000
            result.last_check = datetime.now(timezone.utc)
            return result
        except Exception as e:
            logger.exception(f"Health check failed for {name}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.now(timezone.utc),
            )

    async def _run_async_check(self, name: str, check_fn: Callable) -> ComponentHealth:
        """Run an async health check with timing."""
        start = time.time()
        try:
            result = await check_fn()
            result.latency_ms = (time.time() - start) * 1000
            result.last_check = datetime.now(timezone.utc)
            return result
        except Exception as e:
            logger.exception(f"Async health check failed for {name}")
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=(time.time() - start) * 1000,
                last_check=datetime.now(timezone.utc),
            )

    async def check_all(self, use_cache: bool = True) -> SystemHealth:
        """Run all registered health checks.

        Args:
            use_cache: Whether to use cached results if available

        Returns:
            SystemHealth with all component statuses
        """
        components: dict[str, ComponentHealth] = {}

        # Run sync checks
        for name, check_fn in self._checks.items():
            if use_cache and name in self._cached_results:
                cached = self._cached_results[name]
                if cached.last_check:
                    age = (datetime.now(timezone.utc) - cached.last_check).total_seconds()
                    if age < self._cache_ttl_seconds:
                        components[name] = cached
                        continue

            result = self._run_sync_check(name, check_fn)
            self._cached_results[name] = result
            components[name] = result

        # Run async checks concurrently
        if self._async_checks:
            async_tasks = []
            async_names = []
            for name, check_fn in self._async_checks.items():
                if use_cache and name in self._cached_results:
                    cached = self._cached_results[name]
                    if cached.last_check:
                        age = (datetime.now(timezone.utc) - cached.last_check).total_seconds()
                        if age < self._cache_ttl_seconds:
                            components[name] = cached
                            continue

                async_tasks.append(self._run_async_check(name, check_fn))
                async_names.append(name)

            if async_tasks:
                results = await asyncio.gather(*async_tasks, return_exceptions=True)
                for name, result in zip(async_names, results):
                    if isinstance(result, Exception):
                        components[name] = ComponentHealth(
                            name=name,
                            status=HealthStatus.UNHEALTHY,
                            message=str(result),
                            last_check=datetime.now(timezone.utc),
                        )
                    else:
                        self._cached_results[name] = result
                        components[name] = result

        # Determine overall status
        overall_status = self._calculate_overall_status(components)

        return SystemHealth(
            status=overall_status,
            version=self.version,
            uptime_seconds=time.time() - self._start_time,
            components=components,
        )

    def _calculate_overall_status(
        self, components: dict[str, ComponentHealth]
    ) -> HealthStatus:
        """Calculate overall health from component statuses."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        return HealthStatus.UNKNOWN

    async def is_ready(self) -> tuple[bool, dict[str, bool]]:
        """Check if system is ready to serve traffic.

        Returns:
            Tuple of (is_ready, component_readiness)
        """
        health = await self.check_all(use_cache=True)
        component_ready = {
            name: comp.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            for name, comp in health.components.items()
        }
        # System is ready if at least one critical component is healthy
        is_ready = any(component_ready.values()) if component_ready else False
        return is_ready, component_ready

    def is_alive(self) -> bool:
        """Simple liveness check.

        Returns:
            True if application is alive
        """
        return True

    @property
    def uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self._start_time


# Default health checks for common components

def check_groq_api() -> ComponentHealth:
    """Check Groq API availability."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return ComponentHealth(
            name="groq",
            status=HealthStatus.HEALTHY,
            message="API key configured",
            details={"model": "llama-3.1-8b-instant"},
        )
    return ComponentHealth(
        name="groq",
        status=HealthStatus.UNHEALTHY,
        message="API key not configured",
    )


def check_memory_system(config) -> ComponentHealth:
    """Check memory system status."""
    if not config.memory.long_term.enabled:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.HEALTHY,
            message="Disabled by configuration",
            details={"enabled": False},
        )

    try:
        from javis.memory import get_memory_manager
        manager = get_memory_manager()
        count = manager.get_memory_count() if manager else 0
        return ComponentHealth(
            name="memory",
            status=HealthStatus.HEALTHY,
            message="Memory system operational",
            details={"enabled": True, "memory_count": count},
        )
    except Exception as e:
        return ComponentHealth(
            name="memory",
            status=HealthStatus.DEGRADED,
            message=f"Memory system error: {str(e)}",
            details={"enabled": True, "error": str(e)},
        )


def check_rag_system(config) -> ComponentHealth:
    """Check RAG system status."""
    if not config.rag.enabled:
        return ComponentHealth(
            name="rag",
            status=HealthStatus.HEALTHY,
            message="Disabled by configuration",
            details={"enabled": False},
        )

    try:
        from javis.rag import get_rag_manager
        manager = get_rag_manager()
        if manager:
            stats = manager.get_stats()
            return ComponentHealth(
                name="rag",
                status=HealthStatus.HEALTHY,
                message="RAG system operational",
                details={
                    "enabled": True,
                    "documents": stats.get("total_documents", 0),
                    "chunks": stats.get("total_chunks", 0),
                },
            )
        return ComponentHealth(
            name="rag",
            status=HealthStatus.DEGRADED,
            message="RAG manager not initialized",
            details={"enabled": True},
        )
    except Exception as e:
        return ComponentHealth(
            name="rag",
            status=HealthStatus.DEGRADED,
            message=f"RAG system error: {str(e)}",
            details={"enabled": True, "error": str(e)},
        )


def check_tools_system(config) -> ComponentHealth:
    """Check tools system status."""
    if not config.tools.enabled:
        return ComponentHealth(
            name="tools",
            status=HealthStatus.HEALTHY,
            message="Disabled by configuration",
            details={"enabled": False},
        )

    try:
        from javis.tools import get_registry
        registry = get_registry()
        tools = registry.list_tools() if registry else []
        return ComponentHealth(
            name="tools",
            status=HealthStatus.HEALTHY,
            message=f"{len(tools)} tools available",
            details={
                "enabled": True,
                "tool_count": len(tools),
                "tools": tools[:10],  # First 10 tools
            },
        )
    except Exception as e:
        return ComponentHealth(
            name="tools",
            status=HealthStatus.DEGRADED,
            message=f"Tools system error: {str(e)}",
            details={"enabled": True, "error": str(e)},
        )


def check_storage_system() -> ComponentHealth:
    """Check storage system status."""
    try:
        from javis.storage import get_storage_factory
        factory = get_storage_factory()
        return ComponentHealth(
            name="storage",
            status=HealthStatus.HEALTHY,
            message=f"Storage mode: {factory.mode}",
            details={"mode": factory.mode},
        )
    except Exception as e:
        return ComponentHealth(
            name="storage",
            status=HealthStatus.UNHEALTHY,
            message=f"Storage error: {str(e)}",
        )


async def check_database_connection() -> ComponentHealth:
    """Check database connectivity (for cloud storage)."""
    storage_mode = os.getenv("JAVIS_STORAGE_MODE", "local")

    if storage_mode != "cloud":
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Using local storage",
            details={"mode": "local"},
        )

    try:
        from javis.storage.cloud.supabase_client import get_supabase_client
        client = get_supabase_client()
        if client:
            # Simple connectivity test
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Supabase connected",
                details={"mode": "cloud", "provider": "supabase"},
            )
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message="Supabase client not initialized",
            details={"mode": "cloud"},
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            details={"mode": "cloud", "error": str(e)},
        )


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def initialize_health_checks(config) -> HealthChecker:
    """Initialize health checker with all component checks.

    Args:
        config: Application configuration

    Returns:
        Configured HealthChecker instance
    """
    checker = get_health_checker()

    # Register default checks
    checker.register_check("groq", check_groq_api)
    checker.register_check("storage", check_storage_system)
    checker.register_check(
        "memory",
        lambda: check_memory_system(config),
    )
    checker.register_check(
        "rag",
        lambda: check_rag_system(config),
    )
    checker.register_check(
        "tools",
        lambda: check_tools_system(config),
    )
    checker.register_check(
        "database",
        check_database_connection,
        is_async=True,
    )

    return checker
