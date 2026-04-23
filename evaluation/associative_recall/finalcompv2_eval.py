"""Final composition v2 — evaluation entry point.

Thin wrapper that invokes `final_composition_v2.main`. Kept as a separate
file so the plan's file-naming convention (`finalcompv2_eval.py`) is
respected for operator convenience.

Usage:
    uv run python finalcompv2_eval.py
"""

from final_composition_v2 import main

if __name__ == "__main__":
    main()
