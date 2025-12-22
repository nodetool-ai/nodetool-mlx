"""
System memory utilities for macOS unified memory management.

This module provides best-effort system memory monitoring to prevent
macOS freezes from memory exhaustion. It is conservative and heuristic-based.
"""

import logging
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# Conservative factor for estimating reclaimable inactive memory
# We use 50% of inactive memory as available since not all of it can be reclaimed
INACTIVE_MEMORY_AVAILABLE_FACTOR = 0.5


@dataclass
class SystemMemorySnapshot:
    """Snapshot of system memory state."""

    total_bytes: int
    """Total physical RAM in bytes"""

    available_bytes: int
    """Estimated available memory in bytes (conservative estimate)"""

    free_bytes: int
    """Free memory in bytes"""

    active_bytes: int
    """Active memory in bytes"""

    inactive_bytes: int
    """Inactive memory in bytes"""

    wired_bytes: int
    """Wired (locked) memory in bytes"""

    compressed_bytes: int
    """Compressed memory in bytes"""

    swap_used_bytes: int
    """Swap space in use in bytes"""

    memory_pressure: Optional[str]
    """Memory pressure state: 'normal', 'warn', 'critical', or None if unknown"""

    @property
    def total_gb(self) -> float:
        """Total physical RAM in GB"""
        return self.total_bytes / (1024**3)

    @property
    def available_gb(self) -> float:
        """Estimated available memory in GB"""
        return self.available_bytes / (1024**3)

    @property
    def swap_used_gb(self) -> float:
        """Swap space in use in GB"""
        return self.swap_used_bytes / (1024**3)

    def __str__(self) -> str:
        return (
            f"SystemMemory(total={self.total_gb:.1f}GB, "
            f"available={self.available_gb:.1f}GB, "
            f"swap={self.swap_used_gb:.1f}GB, "
            f"pressure={self.memory_pressure})"
        )


def get_system_memory_snapshot() -> SystemMemorySnapshot:
    """
    Get a conservative snapshot of system memory state on macOS.

    This uses sysctl and vm_stat to estimate available memory.
    The estimate is intentionally conservative for safety.

    Returns:
        SystemMemorySnapshot: Current memory state

    Raises:
        RuntimeError: If not on macOS or if memory detection fails
    """
    if platform.system() != "Darwin":
        raise RuntimeError(
            "System memory monitoring is only supported on macOS (Darwin)"
        )

    try:
        # Get total physical memory
        total_bytes = _get_total_memory_bytes()

        # Get vm_stat for memory breakdown
        vm_stat = _get_vm_stat()

        # Get memory pressure if available
        memory_pressure = _get_memory_pressure()

        # Parse vm_stat
        page_size = vm_stat.get("page_size", 4096)
        free_pages = vm_stat.get("Pages free", 0)
        active_pages = vm_stat.get("Pages active", 0)
        inactive_pages = vm_stat.get("Pages inactive", 0)
        wired_pages = vm_stat.get("Pages wired down", 0)
        compressed_pages = vm_stat.get("Pages occupied by compressor", 0)

        # Calculate bytes
        free_bytes = free_pages * page_size
        active_bytes = active_pages * page_size
        inactive_bytes = inactive_pages * page_size
        wired_bytes = wired_pages * page_size
        compressed_bytes = compressed_pages * page_size

        # Conservative estimate of available memory:
        # Free + some portion of inactive (since it can be reclaimed)
        # We use INACTIVE_MEMORY_AVAILABLE_FACTOR of inactive to be conservative
        available_bytes = free_bytes + int(inactive_bytes * INACTIVE_MEMORY_AVAILABLE_FACTOR)

        # Get swap usage
        swap_used_bytes = _get_swap_used_bytes(vm_stat)

        return SystemMemorySnapshot(
            total_bytes=total_bytes,
            available_bytes=available_bytes,
            free_bytes=free_bytes,
            active_bytes=active_bytes,
            inactive_bytes=inactive_bytes,
            wired_bytes=wired_bytes,
            compressed_bytes=compressed_bytes,
            swap_used_bytes=swap_used_bytes,
            memory_pressure=memory_pressure,
        )

    except Exception as e:
        log.error(f"Failed to get system memory snapshot: {e}")
        raise RuntimeError(f"Failed to get system memory snapshot: {e}") from e


def _get_total_memory_bytes() -> int:
    """Get total physical memory using sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError) as e:
        raise RuntimeError(f"Failed to get total memory: {e}") from e


def _get_vm_stat() -> dict[str, int]:
    """Get vm_stat output parsed into a dictionary."""
    try:
        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        vm_data = {}
        lines = result.stdout.strip().split("\n")

        # First line contains page size
        page_size_match = re.search(r"page size of (\d+) bytes", lines[0])
        if page_size_match:
            vm_data["page_size"] = int(page_size_match.group(1))

        # Parse remaining lines
        for line in lines[1:]:
            # Format: "Pages free:                               12345."
            match = re.match(r"([^:]+):\s+(\d+)", line)
            if match:
                key = match.group(1).strip()
                value = int(match.group(2))
                vm_data[key] = value

        return vm_data

    except subprocess.SubprocessError as e:
        log.warning(f"Failed to get vm_stat: {e}")
        return {"page_size": 4096}  # Default page size


def _get_swap_used_bytes(vm_stat: dict[str, int]) -> int:
    """
    Calculate swap usage from vm_stat.
    
    Note: This is a rough estimate using cumulative swapouts as a proxy
    for swap pressure. It does not represent actual current swap usage,
    which is difficult to measure precisely on macOS. This heuristic
    is conservative and errs on the side of overestimating memory pressure.
    """
    page_size = vm_stat.get("page_size", 4096)
    swapins = vm_stat.get("Swapins", 0)
    swapouts = vm_stat.get("Swapouts", 0)

    # This is a rough estimate - actual swap usage is hard to measure precisely
    # We use swapouts as a proxy for swap pressure
    return swapouts * page_size


def _get_memory_pressure() -> Optional[str]:
    """
    Get memory pressure state using memory_pressure command.

    Returns 'normal', 'warn', 'critical', or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        output = result.stdout.lower()
        if "normal" in output:
            return "normal"
        elif "warn" in output:
            return "warn"
        elif "critical" in output:
            return "critical"

        return None

    except (subprocess.SubprocessError, FileNotFoundError):
        # memory_pressure command may not be available on all macOS versions
        return None


def check_memory_headroom(
    required_bytes: int,
    min_headroom_fraction: float = 0.10,
) -> tuple[bool, SystemMemorySnapshot, int]:
    """
    Check if there's sufficient memory headroom for a job.

    This enforces a conservative rule:
        available_memory >= required_bytes + (min_headroom_fraction * total_memory)

    Args:
        required_bytes: Estimated memory needed for the job
        min_headroom_fraction: Minimum fraction of total memory to keep as headroom
            (default 0.10 = 10%)

    Returns:
        Tuple of (has_sufficient_memory, snapshot, required_headroom_bytes)
    """
    snapshot = get_system_memory_snapshot()

    required_headroom = int(snapshot.total_bytes * min_headroom_fraction)
    total_required = required_bytes + required_headroom

    has_sufficient = snapshot.available_bytes >= total_required

    return has_sufficient, snapshot, required_headroom
