from pipelines.ingest.robots import RobotsGate
import yaml
from pathlib import Path

def main():
    # Load config to get seed URLs and user_agent
    cfg_path = Path("configs/corpus.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    user_agent = cfg.get("user_agent", "RAGScraper/1.0")
    seed_urls = cfg.get("seed_urls", [])

    gate = RobotsGate(user_agent=user_agent)

    print(f"Testing {len(seed_urls)} URLs with user agent: {user_agent}\n")

    for url in seed_urls:
        try:
            allowed = gate.allowed(url)
        except Exception as e:
            allowed = None
            print(f"[ERROR] {url} -> {e}")

        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"{url}\n  -> {status}\n")

if __name__ == "__main__":
    main()
