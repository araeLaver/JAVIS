"""JWT Authentication module for JAVIS API.

This module provides JWT-based authentication for the JAVIS API.
Authentication is optional by default but can be enabled via configuration.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from javis.utils.exceptions import (
    InvalidTokenError,
    MissingTokenError,
    InsufficientPermissionsError,
)

logger = logging.getLogger(__name__)

# Optional JWT dependency - only imported if needed
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None


# ==================== Configuration ====================

class AuthConfig:
    """Authentication configuration from environment variables."""

    # Enable/disable authentication
    AUTH_ENABLED: bool = os.getenv("JAVIS_AUTH_ENABLED", "false").lower() == "true"

    # JWT settings
    JWT_SECRET_KEY: str = os.getenv("JAVIS_JWT_SECRET", "change-this-secret-in-production")
    JWT_ALGORITHM: str = os.getenv("JAVIS_JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JAVIS_JWT_EXPIRATION_HOURS", "24"))

    # API key authentication (simpler alternative)
    API_KEY: Optional[str] = os.getenv("JAVIS_API_KEY")

    # Admin key for privileged operations
    ADMIN_KEY: Optional[str] = os.getenv("JAVIS_ADMIN_KEY")

    @classmethod
    def is_configured(cls) -> bool:
        """Check if authentication is properly configured."""
        if not cls.AUTH_ENABLED:
            return True  # No auth needed
        if cls.API_KEY:
            return True  # API key auth
        if JWT_AVAILABLE and cls.JWT_SECRET_KEY != "change-this-secret-in-production":
            return True  # JWT auth
        return False


# ==================== Models ====================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str = Field(..., description="Subject (user ID)")
    exp: datetime = Field(..., description="Expiration time")
    iat: datetime = Field(..., description="Issued at time")
    role: str = Field(default="user", description="User role")
    permissions: list[str] = Field(default_factory=list, description="Granted permissions")


class TokenResponse(BaseModel):
    """Token response for login endpoint."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class UserInfo(BaseModel):
    """Authenticated user information."""
    user_id: str
    role: str = "user"
    permissions: list[str] = []


# ==================== Security Schemes ====================

# Bearer token authentication
bearer_scheme = HTTPBearer(auto_error=False)


def _extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from request headers."""
    # Check X-API-Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key

    # Check query parameter (less secure, for testing)
    api_key = request.query_params.get("api_key")
    if api_key:
        return api_key

    return None


# ==================== Token Functions ====================

def create_access_token(
    user_id: str,
    role: str = "user",
    permissions: Optional[list[str]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token.

    Args:
        user_id: User identifier
        role: User role (user, admin)
        permissions: List of permissions
        expires_delta: Custom expiration time

    Returns:
        JWT token string

    Raises:
        RuntimeError: If JWT library is not available
    """
    if not JWT_AVAILABLE:
        raise RuntimeError("JWT authentication requires 'pyjwt' package. Install with: pip install pyjwt")

    if expires_delta is None:
        expires_delta = timedelta(hours=AuthConfig.JWT_EXPIRATION_HOURS)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": expire,
        "role": role,
        "permissions": permissions or [],
    }

    token = jwt.encode(
        payload,
        AuthConfig.JWT_SECRET_KEY,
        algorithm=AuthConfig.JWT_ALGORITHM
    )

    return token


def decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Token payload

    Raises:
        InvalidTokenError: If token is invalid or expired
    """
    if not JWT_AVAILABLE:
        raise RuntimeError("JWT authentication requires 'pyjwt' package")

    try:
        payload = jwt.decode(
            token,
            AuthConfig.JWT_SECRET_KEY,
            algorithms=[AuthConfig.JWT_ALGORITHM]
        )
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise InvalidTokenError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise InvalidTokenError(f"Invalid token: {str(e)}")


# ==================== Dependencies ====================

async def get_current_user_optional(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
) -> Optional[UserInfo]:
    """Get current user if authenticated (optional).

    This dependency allows both authenticated and unauthenticated requests.
    Returns None if no authentication is provided.
    """
    # If auth is disabled, return a default user
    if not AuthConfig.AUTH_ENABLED:
        return UserInfo(user_id="anonymous", role="user", permissions=["*"])

    # Try API key authentication first
    api_key = _extract_api_key(request)
    if api_key:
        if api_key == AuthConfig.ADMIN_KEY:
            return UserInfo(user_id="admin", role="admin", permissions=["*"])
        if api_key == AuthConfig.API_KEY:
            return UserInfo(user_id="api-user", role="user", permissions=["*"])
        # Invalid API key - continue to check bearer token

    # Try JWT authentication
    if credentials and credentials.credentials:
        try:
            payload = decode_token(credentials.credentials)
            return UserInfo(
                user_id=payload.sub,
                role=payload.role,
                permissions=payload.permissions,
            )
        except InvalidTokenError:
            pass  # Invalid token, treat as unauthenticated

    return None


async def get_current_user(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)],
) -> UserInfo:
    """Get current user (required).

    This dependency requires authentication. Raises exception if not authenticated.

    Raises:
        MissingTokenError: If no authentication is provided
        InvalidTokenError: If token is invalid
    """
    # If auth is disabled, return a default user
    if not AuthConfig.AUTH_ENABLED:
        return UserInfo(user_id="anonymous", role="user", permissions=["*"])

    # Try API key authentication first
    api_key = _extract_api_key(request)
    if api_key:
        if api_key == AuthConfig.ADMIN_KEY:
            return UserInfo(user_id="admin", role="admin", permissions=["*"])
        if api_key == AuthConfig.API_KEY:
            return UserInfo(user_id="api-user", role="user", permissions=["*"])
        raise InvalidTokenError("Invalid API key")

    # Try JWT authentication
    if not credentials or not credentials.credentials:
        raise MissingTokenError()

    try:
        payload = decode_token(credentials.credentials)
        return UserInfo(
            user_id=payload.sub,
            role=payload.role,
            permissions=payload.permissions,
        )
    except InvalidTokenError:
        raise


async def get_admin_user(
    user: Annotated[UserInfo, Depends(get_current_user)],
) -> UserInfo:
    """Get current user and verify admin role.

    Raises:
        InsufficientPermissionsError: If user is not an admin
    """
    if user.role != "admin":
        raise InsufficientPermissionsError("Admin access required")
    return user


def require_permission(permission: str):
    """Dependency factory to require specific permission.

    Usage:
        @app.get("/protected")
        async def protected(user: Annotated[UserInfo, Depends(require_permission("read:data"))]):
            ...
    """
    async def permission_checker(
        user: Annotated[UserInfo, Depends(get_current_user)],
    ) -> UserInfo:
        if "*" in user.permissions or permission in user.permissions:
            return user
        raise InsufficientPermissionsError(f"Permission required: {permission}")

    return permission_checker


# ==================== Exception Handlers ====================

def auth_exception_handler(request: Request, exc: Exception):
    """Convert auth exceptions to HTTP responses."""
    from fastapi.responses import JSONResponse

    if isinstance(exc, InvalidTokenError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=exc.to_dict(),
            headers={"WWW-Authenticate": "Bearer"},
        )
    if isinstance(exc, MissingTokenError):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=exc.to_dict(),
            headers={"WWW-Authenticate": "Bearer"},
        )
    if isinstance(exc, InsufficientPermissionsError):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content=exc.to_dict(),
        )

    # Re-raise unknown exceptions
    raise exc


# ==================== Utility Functions ====================

def get_auth_status() -> dict:
    """Get current authentication configuration status."""
    return {
        "enabled": AuthConfig.AUTH_ENABLED,
        "jwt_available": JWT_AVAILABLE,
        "api_key_configured": bool(AuthConfig.API_KEY),
        "admin_key_configured": bool(AuthConfig.ADMIN_KEY),
        "properly_configured": AuthConfig.is_configured(),
    }
