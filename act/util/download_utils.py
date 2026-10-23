#===- act/util/download_utils.py - Unavailable download sources --------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   The "unavailable" download status shared by the data loaders and CLIs.
#   A loader returns it when every attempt failed because the remote source
#   could not be reached, after removing what the attempt created; a CLI
#   reports it as a skipped download and exits 0, so an outage of a
#   third-party host does not fail a download command or a CI job.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import errno
import os
import socket
import urllib.error

from act.util.format_utils import rule

__all__ = ["UNAVAILABLE", "is_source_unreachable", "report_unavailable"]

UNAVAILABLE = "unavailable"

_OUTAGE_HTTP_CODES = frozenset({502, 503, 504})
_UNREACHABLE_ERRNOS = frozenset({errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTDOWN, errno.EHOSTUNREACH})


def is_source_unreachable(exc: BaseException) -> bool:
    """Return True if ``exc`` means the download source could not be reached.

    Walks ``exc``, the ``reason`` of a ``urllib.error.URLError`` and the
    cause/context chain. DNS failures, refused/reset/aborted connections,
    timeouts, unreachable networks or hosts and HTTP 502/503/504 count as an
    outage. Any other HTTP status means the source answered (404: the file is
    gone), so it stays a real error.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        err = pending.pop()
        if id(err) in seen:
            continue
        seen.add(id(err))
        if isinstance(err, urllib.error.HTTPError):
            return err.code in _OUTAGE_HTTP_CODES
        if isinstance(err, (ConnectionError, TimeoutError, socket.gaierror)):
            return True
        if isinstance(err, OSError) and err.errno in _UNREACHABLE_ERRNOS:
            return True
        pending.extend(
            nxt for nxt in (getattr(err, "reason", None), err.__cause__, err.__context__)
            if isinstance(nxt, BaseException)
        )
    return False


def report_unavailable(name: str, message: str) -> None:
    """Print a skipped-download notice; under GitHub Actions also emit a warning annotation."""
    print(f"\n{rule(100)}")
    print(f"⚠ SKIPPED: {name} - download source unavailable")
    print(f"  {message}")
    print("  Run the download again once the source is reachable.")
    print(f"{rule(100)}\n")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        text = f"{name}: {message}".replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::warning title=Download skipped::{text}")
