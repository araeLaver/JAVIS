"""Tests for JAVIS service container."""

import pytest
import threading
from unittest.mock import MagicMock, patch

from javis.core.container import (
    ServiceContainer,
    get_container,
    reset_container,
    bootstrap_services,
)


class TestServiceContainer:
    """Tests for ServiceContainer class."""

    def setup_method(self):
        """Create fresh container for each test."""
        self.container = ServiceContainer()

    def test_init_creates_empty_container(self):
        """Test that new container has no services."""
        assert len(self.container.registered_services) == 0

    def test_register_singleton(self):
        """Test registering a singleton service."""
        service = MagicMock()
        self.container.register_singleton("test_service", service)

        assert self.container.has("test_service")
        assert self.container.get("test_service") is service

    def test_register_singleton_overwrites_existing(self):
        """Test that registering same name overwrites previous."""
        service1 = MagicMock()
        service2 = MagicMock()

        self.container.register_singleton("test", service1)
        self.container.register_singleton("test", service2)

        assert self.container.get("test") is service2

    def test_register_factory(self):
        """Test registering a factory function."""
        mock_service = MagicMock()
        factory = MagicMock(return_value=mock_service)

        self.container.register_factory("lazy_service", factory)

        assert self.container.has("lazy_service")
        # Factory not called until service is requested
        factory.assert_not_called()

    def test_factory_called_on_first_get(self):
        """Test that factory is called on first get."""
        mock_service = MagicMock()
        factory = MagicMock(return_value=mock_service)

        self.container.register_factory("lazy_service", factory)
        result = self.container.get("lazy_service")

        factory.assert_called_once()
        assert result is mock_service

    def test_factory_called_only_once(self):
        """Test that factory is called only once (result cached)."""
        mock_service = MagicMock()
        factory = MagicMock(return_value=mock_service)

        self.container.register_factory("lazy_service", factory)

        # Get multiple times
        result1 = self.container.get("lazy_service")
        result2 = self.container.get("lazy_service")
        result3 = self.container.get("lazy_service")

        # Factory called only once
        factory.assert_called_once()
        assert result1 is result2 is result3

    def test_get_raises_keyerror_for_unknown(self):
        """Test that get raises KeyError for unknown service."""
        with pytest.raises(KeyError) as exc_info:
            self.container.get("unknown_service")

        assert "unknown_service" in str(exc_info.value)

    def test_get_optional_returns_service(self):
        """Test get_optional returns service when found."""
        service = MagicMock()
        self.container.register_singleton("test", service)

        result = self.container.get_optional("test")

        assert result is service

    def test_get_optional_returns_none_for_unknown(self):
        """Test get_optional returns None for unknown service."""
        result = self.container.get_optional("unknown")

        assert result is None

    def test_has_returns_true_for_singleton(self):
        """Test has returns True for registered singleton."""
        self.container.register_singleton("test", MagicMock())

        assert self.container.has("test") is True

    def test_has_returns_true_for_factory(self):
        """Test has returns True for registered factory."""
        self.container.register_factory("test", lambda: MagicMock())

        assert self.container.has("test") is True

    def test_has_returns_false_for_unknown(self):
        """Test has returns False for unknown service."""
        assert self.container.has("unknown") is False

    def test_remove_singleton(self):
        """Test removing a singleton service."""
        self.container.register_singleton("test", MagicMock())

        self.container.remove("test")

        assert self.container.has("test") is False

    def test_remove_factory(self):
        """Test removing a factory registration."""
        self.container.register_factory("test", lambda: MagicMock())

        self.container.remove("test")

        assert self.container.has("test") is False

    def test_remove_instantiated_factory(self):
        """Test removing an instantiated factory service."""
        self.container.register_factory("test", lambda: MagicMock())
        self.container.get("test")  # Instantiate

        self.container.remove("test")

        assert self.container.has("test") is False

    def test_remove_unknown_does_not_raise(self):
        """Test removing unknown service doesn't raise."""
        self.container.remove("unknown")  # Should not raise

    def test_reset_clears_services_but_keeps_factories(self):
        """Test reset clears services but keeps factory registrations."""
        factory = MagicMock(return_value=MagicMock())
        self.container.register_singleton("singleton", MagicMock())
        self.container.register_factory("factory_service", factory)
        self.container.get("factory_service")  # Instantiate

        self.container.reset()

        # Singleton should be gone
        assert "singleton" not in self.container._services
        # Factory registration still exists
        assert "factory_service" in self.container._factories
        # But instantiated service is cleared
        assert "factory_service" not in self.container._services

    def test_reset_all_clears_everything(self):
        """Test reset_all clears both services and factories."""
        self.container.register_singleton("singleton", MagicMock())
        self.container.register_factory("factory_service", lambda: MagicMock())

        self.container.reset_all()

        assert len(self.container._services) == 0
        assert len(self.container._factories) == 0
        assert len(self.container.registered_services) == 0

    def test_registered_services_includes_singletons(self):
        """Test registered_services includes singletons."""
        self.container.register_singleton("service1", MagicMock())
        self.container.register_singleton("service2", MagicMock())

        services = self.container.registered_services

        assert "service1" in services
        assert "service2" in services

    def test_registered_services_includes_factories(self):
        """Test registered_services includes factories."""
        self.container.register_factory("lazy1", lambda: MagicMock())
        self.container.register_factory("lazy2", lambda: MagicMock())

        services = self.container.registered_services

        assert "lazy1" in services
        assert "lazy2" in services

    def test_registered_services_no_duplicates(self):
        """Test registered_services doesn't duplicate if factory was instantiated."""
        self.container.register_factory("service", lambda: MagicMock())
        self.container.get("service")  # Instantiate (adds to _services)

        services = self.container.registered_services

        assert services.count("service") == 1


class TestContainerThreadSafety:
    """Tests for container thread safety."""

    def test_concurrent_singleton_registration(self):
        """Test concurrent singleton registrations don't cause issues."""
        container = ServiceContainer()
        errors = []

        def register_service(n):
            try:
                for i in range(100):
                    container.register_singleton(f"service_{n}_{i}", MagicMock())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_service, args=(n,)) for n in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(container.registered_services) == 500

    def test_concurrent_factory_instantiation(self):
        """Test concurrent factory instantiation is thread-safe."""
        container = ServiceContainer()
        call_count = {"count": 0}
        lock = threading.Lock()

        def factory():
            with lock:
                call_count["count"] += 1
            return MagicMock()

        container.register_factory("shared_service", factory)
        results = []

        def get_service():
            result = container.get("shared_service")
            results.append(result)

        threads = [threading.Thread(target=get_service) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Factory called only once due to thread safety
        assert call_count["count"] == 1
        # All threads got the same instance
        assert all(r is results[0] for r in results)


class TestGetContainer:
    """Tests for get_container function."""

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_get_container_returns_container(self):
        """Test get_container returns a ServiceContainer."""
        container = get_container()

        assert isinstance(container, ServiceContainer)

    def test_get_container_returns_same_instance(self):
        """Test get_container returns the same singleton."""
        container1 = get_container()
        container2 = get_container()

        assert container1 is container2

    def test_get_container_thread_safe(self):
        """Test get_container is thread-safe."""
        containers = []

        def get_and_store():
            containers.append(get_container())

        threads = [threading.Thread(target=get_and_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same container
        assert all(c is containers[0] for c in containers)


class TestResetContainer:
    """Tests for reset_container function."""

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_reset_container_clears_services(self):
        """Test reset_container clears all services."""
        container = get_container()
        container.register_singleton("test", MagicMock())

        reset_container()

        new_container = get_container()
        assert not new_container.has("test")

    def test_reset_container_creates_new_instance(self):
        """Test reset_container creates a new container instance."""
        container1 = get_container()

        reset_container()

        container2 = get_container()
        assert container1 is not container2

    def test_reset_container_when_none(self):
        """Test reset_container works when container is None."""
        reset_container()  # Ensure None
        reset_container()  # Should not raise


class TestBootstrapServices:
    """Tests for bootstrap_services function."""

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_bootstrap_services_returns_container(self):
        """Test bootstrap_services returns a container."""
        with patch("javis.utils.config.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.memory.long_term.enabled = False
            mock_config.rag.enabled = False
            mock_config.tools.enabled = False
            mock_get_config.return_value = mock_config

            container = bootstrap_services()

            assert isinstance(container, ServiceContainer)

    def test_bootstrap_services_uses_provided_config(self):
        """Test bootstrap_services uses provided config."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.get("config") is mock_config

    def test_bootstrap_services_loads_default_config(self):
        """Test bootstrap_services loads default config when not provided."""
        with patch("javis.utils.config.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.memory.long_term.enabled = False
            mock_config.rag.enabled = False
            mock_config.tools.enabled = False
            mock_get_config.return_value = mock_config

            container = bootstrap_services()

            mock_get_config.assert_called_once()
            assert container.get("config") is mock_config

    def test_bootstrap_registers_storage_factory(self):
        """Test bootstrap_services registers storage_factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.has("storage_factory")

    def test_bootstrap_registers_store_factories(self):
        """Test bootstrap_services registers conversation, feedback, correction stores."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.has("conversation_store")
        assert container.has("feedback_store")
        assert container.has("correction_store")

    def test_bootstrap_registers_memory_manager(self):
        """Test bootstrap_services registers memory_manager factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.has("memory_manager")

    def test_bootstrap_memory_manager_returns_none_when_disabled(self):
        """Test memory_manager returns None when disabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.get("memory_manager") is None

    def test_bootstrap_memory_manager_when_enabled(self):
        """Test memory_manager is created when enabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = True
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        with patch("javis.memory.get_memory_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager

            container = bootstrap_services(config=mock_config)
            result = container.get("memory_manager")

            mock_get.assert_called_once()
            assert result is mock_manager

    def test_bootstrap_registers_rag_manager(self):
        """Test bootstrap_services registers rag_manager factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.has("rag_manager")

    def test_bootstrap_rag_manager_returns_none_when_disabled(self):
        """Test rag_manager returns None when disabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.get("rag_manager") is None

    def test_bootstrap_rag_manager_when_enabled(self):
        """Test rag_manager is created when enabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = True
        mock_config.tools.enabled = False

        with patch("javis.rag.get_rag_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager

            container = bootstrap_services(config=mock_config)
            result = container.get("rag_manager")

            mock_get.assert_called_once()
            assert result is mock_manager

    def test_bootstrap_registers_tool_registry(self):
        """Test bootstrap_services registers tool_registry factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.has("tool_registry")

    def test_bootstrap_tool_registry_returns_none_when_disabled(self):
        """Test tool_registry returns None when disabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.get("tool_registry") is None

    def test_bootstrap_tool_registry_when_enabled(self):
        """Test tool_registry is created when enabled."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = True

        with patch("javis.tools.get_registry") as mock_get:
            mock_registry = MagicMock()
            mock_get.return_value = mock_registry

            container = bootstrap_services(config=mock_config)
            result = container.get("tool_registry")

            mock_get.assert_called_once()
            assert result is mock_registry

    def test_bootstrap_registers_health_checker(self):
        """Test bootstrap_services registers health_checker factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        container = bootstrap_services(config=mock_config)

        assert container.has("health_checker")

    def test_bootstrap_health_checker_instantiation(self):
        """Test health_checker factory creates health checker."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        with patch("javis.utils.health.get_health_checker") as mock_get:
            mock_checker = MagicMock()
            mock_get.return_value = mock_checker

            container = bootstrap_services(config=mock_config)
            result = container.get("health_checker")

            mock_get.assert_called_once()
            assert result is mock_checker

    def test_bootstrap_storage_factory_instantiation(self):
        """Test storage_factory factory creates StorageFactory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        with patch("javis.storage.factory.StorageFactory") as mock_class:
            mock_factory = MagicMock()
            mock_class.return_value = mock_factory

            container = bootstrap_services(config=mock_config)
            result = container.get("storage_factory")

            mock_class.assert_called_once()
            assert result is mock_factory

    def test_bootstrap_conversation_store_uses_factory(self):
        """Test conversation_store uses storage_factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        mock_store = MagicMock()
        mock_factory = MagicMock()
        mock_factory.get_conversation_store.return_value = mock_store

        with patch("javis.storage.factory.StorageFactory", return_value=mock_factory):
            container = bootstrap_services(config=mock_config)
            result = container.get("conversation_store")

            mock_factory.get_conversation_store.assert_called_once()
            assert result is mock_store

    def test_bootstrap_feedback_store_uses_factory(self):
        """Test feedback_store uses storage_factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        mock_store = MagicMock()
        mock_factory = MagicMock()
        mock_factory.get_feedback_store.return_value = mock_store

        with patch("javis.storage.factory.StorageFactory", return_value=mock_factory):
            container = bootstrap_services(config=mock_config)
            result = container.get("feedback_store")

            mock_factory.get_feedback_store.assert_called_once()
            assert result is mock_store

    def test_bootstrap_correction_store_uses_factory(self):
        """Test correction_store uses storage_factory."""
        mock_config = MagicMock()
        mock_config.memory.long_term.enabled = False
        mock_config.rag.enabled = False
        mock_config.tools.enabled = False

        mock_store = MagicMock()
        mock_factory = MagicMock()
        mock_factory.get_correction_store.return_value = mock_store

        with patch("javis.storage.factory.StorageFactory", return_value=mock_factory):
            container = bootstrap_services(config=mock_config)
            result = container.get("correction_store")

            mock_factory.get_correction_store.assert_called_once()
            assert result is mock_store


class TestContainerIntegration:
    """Integration tests for container."""

    def teardown_method(self):
        """Reset global container after each test."""
        reset_container()

    def test_factory_can_use_other_services(self):
        """Test factory can access other services from container."""
        container = ServiceContainer()

        config = {"db_url": "postgres://localhost"}
        container.register_singleton("config", config)

        def create_db():
            cfg = container.get("config")
            return {"connection": cfg["db_url"]}

        container.register_factory("database", create_db)

        db = container.get("database")
        assert db["connection"] == "postgres://localhost"

    def test_nested_factory_dependencies(self):
        """Test factories that depend on other factories."""
        container = ServiceContainer()

        # Config
        container.register_singleton("config", {"app": "test"})

        # Factory that depends on config
        def create_logger():
            cfg = container.get("config")
            return {"name": cfg["app"]}

        container.register_factory("logger", create_logger)

        # Factory that depends on logger
        def create_service():
            log = container.get("logger")
            return {"logger": log}

        container.register_factory("service", create_service)

        # Getting service should resolve entire chain
        service = container.get("service")
        assert service["logger"]["name"] == "test"
