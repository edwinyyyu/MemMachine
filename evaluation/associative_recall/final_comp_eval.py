"""Final composition evaluation driver.

Thin launcher that imports `final_composition.main` and runs it. Kept
separate from `final_composition.py` (which holds all variant logic and
markdown rendering) so the eval entry point is stable and the module
itself can be imported without side effects.

Usage:
    uv run python final_comp_eval.py
"""

from __future__ import annotations

from final_composition import main


if __name__ == "__main__":
    main()
