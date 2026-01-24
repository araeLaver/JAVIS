"""FastAPI dependency injection providers.

Centralizes all dependency providers for API endpoints.
Uses FastAPI's Depends() system for clean injection.

Usage:
    from javis.interfaces.dependencies import ConfigDep, SessionManagerDep

    @app.get("/api/example")
    async def example(config: ConfigDep, session_mgr: SessionManagerDep):
        ...
"""

from typing import Annotated, Optional

from fastapi import Depends, HTTPException

from javis.utils.config import Config, get_config


# ==================== Config Dependencies ====================


def get_config_dependency() -> Config:
    """Dependency provider for configuration."""
    return get_config()


ConfigDep = Annotated[Config, Depends(get_config_dependency)]


# ==================== Session Dependencies ====================


def get_session_manager_dependency():
    """Dependency provider for session manager."""
    from javis.core.session import get_session_manager

    return get_session_manager()


SessionManagerDep = Annotated[object, Depends(get_session_manager_dependency)]


# ==================== Storage Dependencies ====================


def get_storage_factory_dependency():
    """Dependency provider for storage factory."""
    from javis.storage.factory import get_storage_factory

    return get_storage_factory()


StorageFactoryDep = Annotated[object, Depends(get_storage_factory_dependency)]


def get_conversation_store_dependency():
    """Dependency provider for conversation store."""
    from javis.storage import get_conversation_store

    return get_conversation_store()


ConversationStoreDep = Annotated[object, Depends(get_conversation_store_dependency)]


def get_feedback_store_dependency():
    """Dependency provider for feedback store."""
    from javis.storage import get_feedback_store

    return get_feedback_store()


FeedbackStoreDep = Annotated[object, Depends(get_feedback_store_dependency)]


def get_correction_store_dependency():
    """Dependency provider for correction store."""
    from javis.storage import get_correction_store

    return get_correction_store()


CorrectionStoreDep = Annotated[object, Depends(get_correction_store_dependency)]


# ==================== Memory Dependencies ====================


def get_memory_manager_dependency(config: ConfigDep):
    """Dependency provider for memory manager.

    Returns None if memory is disabled in config.
    """
    if not config.memory.long_term.enabled:
        return None
    from javis.memory import get_memory_manager

    return get_memory_manager()


MemoryManagerDep = Annotated[Optional[object], Depends(get_memory_manager_dependency)]


def require_memory_manager(memory_manager: MemoryManagerDep):
    """Dependency that requires memory to be enabled."""
    if memory_manager is None:
        raise HTTPException(status_code=400, detail="Memory system is not enabled")
    return memory_manager


RequiredMemoryManagerDep = Annotated[object, Depends(require_memory_manager)]


# ==================== RAG Dependencies ====================


def get_rag_manager_dependency(config: ConfigDep):
    """Dependency provider for RAG manager.

    Returns None if RAG is disabled in config.
    """
    if not config.rag.enabled:
        return None
    from javis.rag import get_rag_manager

    return get_rag_manager()


RAGManagerDep = Annotated[Optional[object], Depends(get_rag_manager_dependency)]


def require_rag_manager(rag_manager: RAGManagerDep):
    """Dependency that requires RAG to be enabled."""
    if rag_manager is None:
        raise HTTPException(status_code=400, detail="RAG system is not enabled")
    return rag_manager


RequiredRAGManagerDep = Annotated[object, Depends(require_rag_manager)]


# ==================== Tools Dependencies ====================


def get_tool_registry_dependency(config: ConfigDep):
    """Dependency provider for tool registry.

    Returns None if tools are disabled in config.
    """
    if not config.tools.enabled:
        return None
    from javis.tools import get_registry

    return get_registry()


ToolRegistryDep = Annotated[Optional[object], Depends(get_tool_registry_dependency)]


def require_tool_registry(tool_registry: ToolRegistryDep):
    """Dependency that requires tools to be enabled."""
    if tool_registry is None:
        raise HTTPException(status_code=400, detail="Tools system is not enabled")
    return tool_registry


RequiredToolRegistryDep = Annotated[object, Depends(require_tool_registry)]


# ==================== Health Dependencies ====================


def get_health_checker_dependency():
    """Dependency provider for health checker."""
    from javis.utils.health import get_health_checker

    return get_health_checker()


HealthCheckerDep = Annotated[object, Depends(get_health_checker_dependency)]


# ==================== Training Dependencies ====================


def get_correction_manager_dependency():
    """Dependency provider for correction manager."""
    from javis.training.corrections import get_correction_manager

    return get_correction_manager()


CorrectionManagerDep = Annotated[object, Depends(get_correction_manager_dependency)]


def get_feedback_manager_dependency():
    """Dependency provider for feedback store (training)."""
    from javis.training.feedback_store import get_feedback_store

    return get_feedback_store()


FeedbackManagerDep = Annotated[object, Depends(get_feedback_manager_dependency)]


# ==================== Model Client Dependencies ====================


class ModelClientProvider:
    """Provider for model clients with caching."""

    _local_client = None
    _modal_client = None

    @classmethod
    def get_local_client(cls, adapter_version: Optional[str] = None):
        """Get or create local model client."""
        from javis.models.local_client import LocalModelClient, get_latest_adapter

        adapter_path = (
            adapter_version
            if adapter_version
            else get_latest_adapter()
        )

        if cls._local_client is None or cls._local_client.adapter_path != adapter_path:
            cls._local_client = LocalModelClient(adapter_path=adapter_path)

        return cls._local_client

    @classmethod
    def get_modal_client(cls, adapter_version: Optional[str] = None):
        """Get or create modal model client."""
        from javis.models.modal_client import (
            ModalInferenceClient,
            get_latest_adapter_path,
        )

        adapter_path = (
            adapter_version
            if adapter_version
            else get_latest_adapter_path()
        )

        # Modal client is created fresh each time for stateless operation
        cls._modal_client = ModalInferenceClient(adapter_path=adapter_path)
        return cls._modal_client

    @classmethod
    def reset(cls):
        """Reset cached clients (for testing)."""
        cls._local_client = None
        cls._modal_client = None


def get_model_client_provider():
    """Dependency provider for model client provider."""
    return ModelClientProvider


ModelClientProviderDep = Annotated[type, Depends(get_model_client_provider)]


# ==================== Container Dependencies ====================


def get_container_dependency():
    """Dependency provider for service container."""
    from javis.core.container import get_container

    return get_container()


ContainerDep = Annotated[object, Depends(get_container_dependency)]
