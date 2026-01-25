"""Service container for dependency injection.

Provides centralized service management with:
- Singleton registration
- Factory registration (lazy instantiation)
- Thread-safe access
- Easy testing support
"""

import logging
import threading
from typing import Any, Callable, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceContainer:
    """Service locator and dependency container.

    Provides centralized management of application services with support for:
    - Singleton services (instantiated once)
    - Factory functions (lazy instantiation)
    - Thread-safe access
    - Easy reset for testing

    Example:
        container = ServiceContainer()
        container.register_singleton("config", Config())
        container.register_factory("db", lambda: Database(container.get("config")))

        config = container.get("config")
        db = container.get("db")
    """

    def __init__(self):
        self._services: dict[str, Any] = {}
        self._factories: dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()  # RLock to allow reentrant calls from factories

    def register_singleton(self, name: str, service: Any) -> None:
        """Register a singleton service instance.

        Args:
            name: Service identifier
            service: Service instance
        """
        with self._lock:
            self._services[name] = service
            logger.debug(f"Registered singleton service: {name}")

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a factory function for lazy instantiation.

        The factory will be called once when the service is first requested.

        Args:
            name: Service identifier
            factory: Factory function that creates the service
        """
        with self._lock:
            self._factories[name] = factory
            logger.debug(f"Registered factory for service: {name}")

    def get(self, name: str) -> Any:
        """Get a service by name.

        If the service was registered as a factory, it will be instantiated
        on first access and cached for subsequent calls.

        Args:
            name: Service identifier

        Returns:
            The service instance

        Raises:
            KeyError: If service is not registered
        """
        with self._lock:
            # Check if already instantiated
            if name in self._services:
                return self._services[name]

            # Check if factory exists
            if name in self._factories:
                logger.debug(f"Instantiating service from factory: {name}")
                self._services[name] = self._factories[name]()
                return self._services[name]

            raise KeyError(f"Service '{name}' not found in container")

    def get_optional(self, name: str) -> Optional[Any]:
        """Get a service by name, returning None if not found.

        Args:
            name: Service identifier

        Returns:
            The service instance or None
        """
        try:
            return self.get(name)
        except KeyError:
            return None

    def has(self, name: str) -> bool:
        """Check if a service is registered.

        Args:
            name: Service identifier

        Returns:
            True if service is registered
        """
        with self._lock:
            return name in self._services or name in self._factories

    def remove(self, name: str) -> None:
        """Remove a service registration.

        Args:
            name: Service identifier
        """
        with self._lock:
            self._services.pop(name, None)
            self._factories.pop(name, None)

    def reset(self) -> None:
        """Reset all services (useful for testing).

        Clears all service instances but keeps factory registrations.
        """
        with self._lock:
            self._services.clear()
            logger.debug("Container services reset")

    def reset_all(self) -> None:
        """Reset all services and factories."""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            logger.debug("Container fully reset")

    @property
    def registered_services(self) -> list[str]:
        """Get list of registered service names."""
        with self._lock:
            return list(set(self._services.keys()) | set(self._factories.keys()))


# Global container instance
_container: Optional[ServiceContainer] = None
_container_lock = threading.RLock()


def get_container() -> ServiceContainer:
    """Get the global service container instance.

    Returns:
        The global ServiceContainer
    """
    global _container
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = ServiceContainer()
    return _container


def reset_container() -> None:
    """Reset the global container (for testing)."""
    global _container
    with _container_lock:
        if _container is not None:
            _container.reset_all()
        _container = None


def bootstrap_services(config: Any = None) -> ServiceContainer:
    """Bootstrap the service container with all application services.

    Args:
        config: Optional configuration object. If None, will load default config.

    Returns:
        The bootstrapped ServiceContainer
    """
    container = get_container()

    # Register config
    if config is None:
        from javis.utils.config import get_config
        config = get_config()
    container.register_singleton("config", config)

    # Register storage factory
    def create_storage_factory():
        from javis.storage.factory import StorageFactory
        return StorageFactory()

    container.register_factory("storage_factory", create_storage_factory)

    # Register individual stores (convenience accessors)
    def create_conversation_store():
        factory = container.get("storage_factory")
        return factory.get_conversation_store()

    def create_feedback_store():
        factory = container.get("storage_factory")
        return factory.get_feedback_store()

    def create_correction_store():
        factory = container.get("storage_factory")
        return factory.get_correction_store()

    container.register_factory("conversation_store", create_conversation_store)
    container.register_factory("feedback_store", create_feedback_store)
    container.register_factory("correction_store", create_correction_store)

    # Register memory manager (if enabled)
    def create_memory_manager():
        cfg = container.get("config")
        if not cfg.memory.long_term.enabled:
            return None
        from javis.memory import get_memory_manager
        return get_memory_manager()

    container.register_factory("memory_manager", create_memory_manager)

    # Register RAG manager (if enabled)
    def create_rag_manager():
        cfg = container.get("config")
        if not cfg.rag.enabled:
            return None
        from javis.rag import get_rag_manager
        return get_rag_manager()

    container.register_factory("rag_manager", create_rag_manager)

    # Register tool registry (if enabled)
    def create_tool_registry():
        cfg = container.get("config")
        if not cfg.tools.enabled:
            return None
        from javis.tools import get_registry
        return get_registry()

    container.register_factory("tool_registry", create_tool_registry)

    # Register health checker
    def create_health_checker():
        from javis.utils.health import get_health_checker
        return get_health_checker()

    container.register_factory("health_checker", create_health_checker)

    logger.info("Service container bootstrapped")
    return container
