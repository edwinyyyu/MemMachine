"""Set vectors.on_disk=True on all longmemeval collections in Qdrant.

Triggers in-place background migration (no reingestion). Skips the registry
collection (longmemeval__registry) since it's tiny and only used for metadata.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models


async def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--namespace",
        default="longmemeval",
        help="Namespace prefix to match (default: longmemeval)",
    )
    parser.add_argument(
        "--on-disk",
        action="store_true",
        help="Set vectors.on_disk=True (default: False for a dry run listing)",
    )
    args = parser.parse_args()

    client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )

    try:
        collections = await client.get_collections()
        prefix = f"{args.namespace}__"
        registry_name = f"{args.namespace}__registry"

        targets = [
            c.name
            for c in collections.collections
            if c.name.startswith(prefix) and c.name != registry_name
        ]

        print(f"Found {len(targets)} collections matching prefix '{prefix}'")

        if not args.on_disk:
            print("Dry run — pass --on-disk to actually update")
            for name in targets[:10]:
                print(f"  {name}")
            if len(targets) > 10:
                print(f"  ... and {len(targets) - 10} more")
            return

        succeeded = 0
        failed: list[tuple[str, str]] = []
        for i, name in enumerate(targets):
            try:
                await client.update_collection(
                    collection_name=name,
                    vectors_config={"": models.VectorParamsDiff(on_disk=True)},
                )
                succeeded += 1
                if (i + 1) % 50 == 0 or i + 1 == len(targets):
                    print(f"  updated {i + 1}/{len(targets)}")
            except Exception as e:
                failed.append((name, str(e)))
                print(f"  FAILED {name}: {e}")

        print(f"\nSucceeded: {succeeded}/{len(targets)}")
        if failed:
            print(f"Failed: {len(failed)}")
            for name, err in failed[:5]:
                print(f"  {name}: {err}")
    finally:
        await client.close()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
