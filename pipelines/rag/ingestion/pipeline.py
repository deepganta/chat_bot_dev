# pipeline.py
import argparse
import logging

import yaml

from .graph_ingest import build_graph

log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to corpus.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    app = build_graph().compile()         # compile the LangGraph into an app
    final_state = app.invoke({"config": cfg})  # run once with initial state
    log.info("Done. Store stats: %s", final_state.get("store_stats"))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    main()
