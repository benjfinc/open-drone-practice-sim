import os
import subprocess
import sys
from pathlib import Path

from fpvsim import gl_platform

ROOT = Path(__file__).resolve().parents[1]


class FakeModernGL:
    def __init__(self):
        self.calls = []

    def create_context(self, **kwargs):
        self.calls.append(("window", kwargs))
        return object()

    def create_standalone_context(self, **kwargs):
        self.calls.append(("standalone", kwargs))
        return object()


def test_windows_uses_native_sdl_and_gl_loader(monkeypatch):
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    gl_platform.configure_environment("nvidia", platform_name="win32")
    assert "SDL_VIDEODRIVER" not in os.environ
    assert "__NV_PRIME_RENDER_OFFLOAD" not in os.environ
    assert os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] == "1"
    fake = FakeModernGL()
    gl_platform.create_window_context(fake, platform_name="win32")
    gl_platform.create_standalone_context(fake, platform_name="win32")
    assert fake.calls == [("window", {}), ("standalone", {})]


def test_linux_keeps_tested_x11_glx_and_egl_path(monkeypatch):
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)
    gl_platform.configure_environment("nvidia", platform_name="linux")
    assert os.environ["SDL_VIDEODRIVER"] == "x11"
    assert os.environ["__NV_PRIME_RENDER_OFFLOAD"] == "1"
    assert os.environ["__GLX_VENDOR_LIBRARY_NAME"] == "nvidia"
    fake = FakeModernGL()
    gl_platform.create_window_context(fake, platform_name="linux")
    gl_platform.create_standalone_context(fake, platform_name="linux")
    assert fake.calls == [
        ("window", {"libgl": "libGL.so.1"}),
        ("standalone", {"backend": "egl"}),
    ]


def test_module_entry_point_lists_tracks():
    result = subprocess.run(
        [sys.executable, "-m", "fpvsim", "--list-tracks"], cwd=ROOT,
        capture_output=True, text=True, timeout=15, check=True)
    assert "neon-circuit" in result.stdout
    assert "open-world" in result.stdout
