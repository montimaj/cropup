"""Build the RAG knowledge cards and their embedding index (SPEC section 6).

Reads only files that are committed under ``cropup/data/`` -- the disease
library and the Earth Engine registry -- plus the practice cards ported into
``cropup/rag/corpus.py`` from the vendored reference backend. ``Data/`` is
gitignored and is not read here, so the corpus rebuilds on a clean checkout.

    python tools/build_corpus.py              # cards + index
    python tools/build_corpus.py --no-index   # cards only (no encoder needed)
    python tools/build_corpus.py --check      # verify the built index is current

**Offline.** Nothing here reaches the network. Earth Engine in particular is
never initialised: this script used to call ``bootstrap.initialize()``, which
opens an ``ee.Initialize`` and, with ``CROPUP_EE_VERIFY_ON_INIT``, a live round
trip -- so a committed artifact built from two local files could not be
reproduced without credentials, and would fail to rebuild in CI. The encoder is
read from the local Hugging Face cache; set ``HF_HUB_OFFLINE=1`` to prove it.

The index step imports ``cropup.nlu.embed``. If that encoder is unavailable the
cards are still written and the exit code is 2: the app can boot without an
index, it just cannot answer knowledge questions until one exists.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from cropup import config  # noqa: E402
from cropup.errors import CropUpError  # noqa: E402
from cropup.rag import corpus, index  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-index", action="store_true", help="write cards only, skip embedding")
    parser.add_argument("--check", action="store_true", help="report whether the built index is current")
    args = parser.parse_args()

    settings = config.get_settings()

    if args.check:
        return _check(settings)

    cards = corpus.build_cards(settings)
    summary = corpus.write_corpus(cards, settings.corpus_dir)
    print(f"wrote {summary['card_count']} cards -> {corpus.cards_path(settings)}")
    for kind, count in summary["by_kind"].items():
        print(f"  {kind:9s} {count}")
    print(f"  crops named: {summary['crops_covered']}")
    print(f"  digest:      {summary['digest'][:16]}")

    if args.no_index:
        return 0

    try:
        built = index.build(cards, settings=settings)
    except CropUpError as exc:
        print(f"index NOT built: {exc}", file=sys.stderr)
        print("cards are written; rerun once cropup.nlu.embed can load.", file=sys.stderr)
        return 2
    print(
        f"wrote index {built.vectors.shape[0]}x{built.vectors.shape[1]} "
        f"({built.model_id}) -> {settings.corpus_dir / index.VECTORS_FILE}"
    )
    return 0


def _check(settings):
    # index.load() reads the cards from cards.jsonl and refuses an index whose
    # count or digest has drifted from them, so loading it *is* the check.
    try:
        built = index.load(settings=settings, refresh=True)
    except CropUpError as exc:
        print(f"corpus not usable: {exc}", file=sys.stderr)
        return 1
    print(
        f"{len(built)} cards, index {built.vectors.shape[0]}x{built.vectors.shape[1]} "
        f"built {built.built_at}"
    )
    print(f"index digest matches {corpus.CARDS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
