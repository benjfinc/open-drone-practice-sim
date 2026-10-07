"""Small platform boundary for SDL and ModernGL context creation."""
from __future__ import annotations

import os
import sys


def configure_environment(gpu="default", platform_name=None):
    """Set only the environment hints appropriate for the current OS."""
    platform_name = platform_name or sys.platform
    if platform_name.startswith("linux"):
        # X11/GLX is the visibly tested Linux path. Respect an explicit caller choice.
        os.environ.setdefault("SDL_VIDEODRIVER", "x11")
        if gpu == "nvidia":
            os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
            os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def create_window_context(moderngl, platform_name=None):
    """Attach ModernGL to pygame's current context using the native GL loader."""
    platform_name = platform_name or sys.platform
    if platform_name.startswith("linux"):
        return moderngl.create_context(libgl="libGL.so.1")
    return moderngl.create_context()


def create_standalone_context(moderngl, platform_name=None):
    """Create an offscreen context for benchmarks on supported desktop platforms."""
    platform_name = platform_name or sys.platform
    if platform_name.startswith("linux"):
        return moderngl.create_standalone_context(backend="egl")
    return moderngl.create_standalone_context()
