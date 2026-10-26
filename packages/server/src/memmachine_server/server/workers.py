"""The number of uvicorn worker processes the server runs."""

import logging
import os

logger = logging.getLogger(__name__)


def worker_count() -> int:
    """Resolve MEMMACHINE_WORKERS, defaulting to 1.

    Note: We do not use (os.cpu_count() - 1) as this is often inaccurate in
    container environments (reporting host CPUs vs container limits). We leave
    it to the user to configure MEMMACHINE_WORKERS based on their allocated
    vCPUs to avoid creating excessive worker processes at startup.
    """
    workers_env = os.getenv("MEMMACHINE_WORKERS")
    if not workers_env:
        return 1
    try:
        return int(workers_env)
    except ValueError:
        logger.warning(
            "Invalid MEMMACHINE_WORKERS value '%s'. Defaulting to 1.", workers_env
        )
        return 1
