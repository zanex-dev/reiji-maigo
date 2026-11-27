"""Reiji-Maigo Memory Swarm infrastructure (runner, guards, workers, coordinator)."""

from __future__ import annotations

from reiji_maigo.swarm.archivist import (
    ARCHIVIST_L0_PREFIX,
    ArchivistAgent,
    ArchivistReport,
)
from reiji_maigo.swarm.coordinator import SwarmCoordinator
from reiji_maigo.swarm.guards import (
    ArchivistCapabilities,
    ScoutCapabilities,
    ScribeCapabilities,
    build_role_capabilities,
    ensure_swarm_tables,
    get_pool_event_note,
    list_swarm_telemetry,
    pop_pool_event_note,
    record_swarm_telemetry,
    save_pool_event_note,
)
from reiji_maigo.swarm.runner import (
    DEFAULT_FLASH_MODEL,
    ENV_FLASH_API_KEY,
    ENV_FLASH_BASE_URL,
    ENV_FLASH_MODEL,
    CallableFlashRunner,
    HttpFlashRunner,
    SwarmCallResult,
    SwarmJSONError,
    SwarmModelRunner,
    SwarmRunnerError,
    compute_prefix_sha256,
    extract_json_object,
)
from reiji_maigo.swarm.scout import (
    SCOUT_L0_PREFIX,
    ScoutAgent,
    ScoutBriefing,
    estimate_tokens,
)
from reiji_maigo.swarm.scribe import (
    DEFAULT_WATERMARK_KEY,
    SCRIBE_L0_PREFIX,
    AsyncScribeWorker,
    ScribeAgent,
    ScribeDigestReport,
)

__all__ = [
    "ARCHIVIST_L0_PREFIX",
    "ArchivistAgent",
    "ArchivistCapabilities",
    "ArchivistReport",
    "AsyncScribeWorker",
    "CallableFlashRunner",
    "DEFAULT_FLASH_MODEL",
    "DEFAULT_WATERMARK_KEY",
    "ENV_FLASH_API_KEY",
    "ENV_FLASH_BASE_URL",
    "ENV_FLASH_MODEL",
    "HttpFlashRunner",
    "SCOUT_L0_PREFIX",
    "SCRIBE_L0_PREFIX",
    "ScoutAgent",
    "ScoutBriefing",
    "ScoutCapabilities",
    "ScribeAgent",
    "ScribeCapabilities",
    "ScribeDigestReport",
    "SwarmCallResult",
    "SwarmCoordinator",
    "SwarmJSONError",
    "SwarmModelRunner",
    "SwarmRunnerError",
    "build_role_capabilities",
    "compute_prefix_sha256",
    "ensure_swarm_tables",
    "estimate_tokens",
    "extract_json_object",
    "get_pool_event_note",
    "list_swarm_telemetry",
    "pop_pool_event_note",
    "record_swarm_telemetry",
    "save_pool_event_note",
]
