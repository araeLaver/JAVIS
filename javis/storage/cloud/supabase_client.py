"""Supabase client wrapper for JAVIS cloud storage."""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Wrapper for Supabase client with connection management."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """Initialize Supabase client.

        Args:
            url: Supabase project URL. Defaults to SUPABASE_URL env var.
            key: Supabase service role key. Defaults to SUPABASE_SERVICE_ROLE_KEY env var.
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not self.url or not self.key:
            raise ValueError(
                "Supabase URL and key must be provided. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables."
            )

        self._client = None

    @property
    def client(self) -> Any:
        """Get or create Supabase client (lazy initialization)."""
        if self._client is None:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
                logger.info("Supabase client connected successfully")
            except ImportError:
                raise ImportError(
                    "supabase package is required for cloud storage. "
                    "Install it with: pip install supabase"
                )
            except Exception as e:
                logger.error(f"Failed to connect to Supabase: {e}")
                raise

        return self._client

    def table(self, name: str) -> Any:
        """Get a table reference.

        Args:
            name: Table name

        Returns:
            Supabase table reference
        """
        return self.client.table(name)

    def insert(self, table: str, data: dict) -> dict:
        """Insert a record.

        Args:
            table: Table name
            data: Record data

        Returns:
            Inserted record
        """
        result = self.table(table).insert(data).execute()
        if result.data:
            return result.data[0]
        raise Exception(f"Insert failed: {result}")

    def upsert(self, table: str, data: dict, on_conflict: str = "id") -> dict:
        """Upsert a record.

        Args:
            table: Table name
            data: Record data
            on_conflict: Conflict column

        Returns:
            Upserted record
        """
        result = self.table(table).upsert(data, on_conflict=on_conflict).execute()
        if result.data:
            return result.data[0]
        raise Exception(f"Upsert failed: {result}")

    def select(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[dict] = None,
        order: Optional[str] = None,
        desc: bool = True,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> list[dict]:
        """Select records with optional filters.

        Args:
            table: Table name
            columns: Columns to select
            filters: Filter conditions as {column: value} or {column: {"op": operator, "val": value}}
            order: Column to order by
            desc: Whether to order descending
            limit: Maximum records to return
            offset: Offset for pagination

        Returns:
            List of records
        """
        query = self.table(table).select(columns)

        if filters:
            for column, condition in filters.items():
                if isinstance(condition, dict):
                    op = condition.get("op", "eq")
                    val = condition.get("val")
                    if op == "eq":
                        query = query.eq(column, val)
                    elif op == "neq":
                        query = query.neq(column, val)
                    elif op == "gt":
                        query = query.gt(column, val)
                    elif op == "gte":
                        query = query.gte(column, val)
                    elif op == "lt":
                        query = query.lt(column, val)
                    elif op == "lte":
                        query = query.lte(column, val)
                    elif op == "like":
                        query = query.like(column, val)
                    elif op == "ilike":
                        query = query.ilike(column, val)
                    elif op == "in":
                        query = query.in_(column, val)
                else:
                    query = query.eq(column, condition)

        if order:
            query = query.order(order, desc=desc)

        if limit:
            query = query.limit(limit)

        if offset:
            query = query.offset(offset)

        result = query.execute()
        return result.data or []

    def select_one(
        self,
        table: str,
        columns: str = "*",
        filters: Optional[dict] = None,
    ) -> Optional[dict]:
        """Select a single record.

        Args:
            table: Table name
            columns: Columns to select
            filters: Filter conditions

        Returns:
            Record or None
        """
        records = self.select(table, columns, filters, limit=1)
        return records[0] if records else None

    def update(
        self,
        table: str,
        data: dict,
        filters: dict,
    ) -> list[dict]:
        """Update records matching filters.

        Args:
            table: Table name
            data: Data to update
            filters: Filter conditions

        Returns:
            Updated records
        """
        query = self.table(table).update(data)

        for column, value in filters.items():
            query = query.eq(column, value)

        result = query.execute()
        return result.data or []

    def delete(
        self,
        table: str,
        filters: dict,
    ) -> list[dict]:
        """Delete records matching filters.

        Args:
            table: Table name
            filters: Filter conditions

        Returns:
            Deleted records
        """
        query = self.table(table).delete()

        for column, value in filters.items():
            query = query.eq(column, value)

        result = query.execute()
        return result.data or []

    def count(
        self,
        table: str,
        filters: Optional[dict] = None,
    ) -> int:
        """Count records.

        Args:
            table: Table name
            filters: Filter conditions

        Returns:
            Record count
        """
        query = self.table(table).select("*", count="exact")

        if filters:
            for column, value in filters.items():
                query = query.eq(column, value)

        result = query.execute()
        return result.count or 0


# Global client instance
_client: Optional[SupabaseClient] = None


def get_supabase_client() -> SupabaseClient:
    """Get or create the global Supabase client.

    Returns:
        SupabaseClient instance
    """
    global _client
    if _client is None:
        _client = SupabaseClient()
    return _client


def reset_supabase_client() -> None:
    """Reset the global Supabase client. Useful for testing."""
    global _client
    _client = None
