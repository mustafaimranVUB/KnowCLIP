"""Unit tests for _resource_snapshot() — must pass on CPU without GPU or pynvml."""
import pytest


def test_resource_snapshot_returns_dict():
    from backend.app import _resource_snapshot

    snap = _resource_snapshot()
    assert isinstance(snap, dict)


def test_resource_snapshot_key_types():
    from backend.app import _resource_snapshot

    snap = _resource_snapshot()
    for key, value in snap.items():
        assert isinstance(key, str), f"Key {key!r} is not a str"
        assert isinstance(value, (int, float, str)), f"Value for {key!r} has unexpected type {type(value)}"


def test_resource_snapshot_psutil_fields():
    """When psutil is installed the CPU/RAM fields must be positive numbers."""
    pytest.importorskip("psutil")
    from backend.app import _resource_snapshot

    snap = _resource_snapshot()
    assert "cpu_percent" in snap
    assert "ram_used_mb" in snap
    assert "ram_total_mb" in snap
    assert "process_rss_mb" in snap
    assert snap["ram_used_mb"] > 0
    assert snap["ram_total_mb"] > 0
    assert snap["process_rss_mb"] > 0
    assert 0.0 <= snap["cpu_percent"] <= 100.0


def test_resource_snapshot_no_gpu_on_cpu():
    """On a CPU-only machine no GPU keys should appear."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("GPU present — skipping CPU-only assertion")

    from backend.app import _resource_snapshot

    snap = _resource_snapshot()
    assert "gpu_name" not in snap
    assert "gpu_mem_allocated_mb" not in snap
    assert "gpu_util_pct" not in snap
