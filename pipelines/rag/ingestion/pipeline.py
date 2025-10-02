# pipeline.py
import argparse, yaml
from .graph_ingest import build_graph

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to corpus.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    app = build_graph().compile()         # compile the LangGraph into an app
    final_state = app.invoke({"config": cfg})  # run once with initial state
    print("Done. Store stats:", final_state.get("store_stats"))

if __name__ == "__main__":
    main()
