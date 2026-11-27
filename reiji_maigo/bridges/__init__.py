"""Optional duck-typed bridges to external host runtimes."""

from __future__ import annotations

from reiji_maigo.bridges.shinobu import ReijiShinobuPlugin, create_plugin

__all__ = [
    "ReijiShinobuPlugin",
    "create_plugin",
]
