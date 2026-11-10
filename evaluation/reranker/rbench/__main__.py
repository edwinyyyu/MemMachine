import errno
import shutil
import sqlite3

from rbench import limits
from rbench.cli import main
from rbench.credits import OutOfCredits
from rbench.paths import DATA


def _disk_full(err: BaseException) -> bool:
    if isinstance(err, OSError) and err.errno == errno.ENOSPC:
        return True
    return isinstance(err, sqlite3.OperationalError) and "full" in str(err).lower()


try:
    main()
except (OutOfCredits, limits.GaveUp, limits.Interrupted) as err:  # finished work is kept: the message says how to resume
    raise SystemExit(str(err)) from None
except (OSError, sqlite3.OperationalError) as err:
    if not _disk_full(err):
        raise
    where = getattr(err, "filename", None) or DATA
    free = shutil.disk_usage(DATA).free / 1e9
    raise SystemExit(f"disk full at {where} ({free:.1f} GB free): free space and rerun; finished work is kept") from None
