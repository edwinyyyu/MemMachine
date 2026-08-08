#!/usr/bin/env python3
"""Optionally rename artifacts to their canonical names. Fully reversible.

    python3 apply_rename.py --dry-run
    python3 apply_rename.py --apply     # writes rename_undo.json
    python3 apply_rename.py --undo

WARNING: 52 scripts in this directory hardcode 430 artifact filenames, and
scripts derive names from each other (f"search-{tag}.json" -> eval, sqlite).
Renaming breaks those references and makes past runs unreproducible. Prefer
the `_by_config/` symlink index unless these runs are retired. See NAMING-v2.md.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "rename_plan.json")
UNDO = os.path.join(HERE, "rename_undo.json")


def load(p):
    with open(p) as f:
        return json.load(f)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--dry-run"

    if mode == "--undo":
        if not os.path.exists(UNDO):
            sys.exit("no rename_undo.json — nothing to undo")
        done = 0
        for e in reversed(load(UNDO)):
            new, old = os.path.join(HERE, e["to"]), os.path.join(HERE, e["from"])
            if os.path.exists(new) and not os.path.exists(old):
                os.rename(new, old)
                done += 1
        print(f"restored {done} original names")
        return

    if not os.path.exists(PLAN):
        sys.exit("no rename_plan.json — run manifest.py first")
    plan = load(PLAN)

    todo, missing, clash = [], 0, 0
    for e in plan:
        src = os.path.join(HERE, e["original"])
        dst = os.path.join(HERE, e["canonical"])
        if not os.path.isfile(src):
            missing += 1
            continue
        if os.path.exists(dst):
            clash += 1
            continue
        todo.append(e)

    print(f"plan: {len(plan)}  renameable: {len(todo)}  missing: {missing}  "
          f"would-clash: {clash}")

    if mode == "--dry-run":
        for e in todo[:15]:
            print(f"  {e['original'][:58]}\n    -> {e['canonical'][:88]}")
        print("  ... (--apply to execute)")
        return

    if mode != "--apply":
        sys.exit("use --dry-run, --apply or --undo")

    done = []
    for e in todo:
        os.rename(os.path.join(HERE, e["original"]),
                  os.path.join(HERE, e["canonical"]))
        done.append({"from": e["original"], "to": e["canonical"]})
    with open(UNDO, "w") as f:
        json.dump(done, f, indent=1)
    print(f"renamed {len(done)} files; undo log -> rename_undo.json")


if __name__ == "__main__":
    main()
