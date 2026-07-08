"""``python -m rhob`` -- dispatches to the v3 CLI (``rhob.v3.cli``)."""

from __future__ import annotations

import sys

from rhob.v3.cli.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
