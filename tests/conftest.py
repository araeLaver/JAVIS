"""Pytest configuration and fixtures."""

import os

# Disable rate limiting before any module imports
# This must be at module level to take effect before app initialization
os.environ["JAVIS_RATE_LIMIT_ENABLED"] = "false"
