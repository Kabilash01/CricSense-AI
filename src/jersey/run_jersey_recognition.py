import sys
from pathlib import Path

# --------------------------------------------------
# Add src/ to PYTHONPATH
# --------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jersey.smolvlm_reader import SmolVLMJerseyReader
from jersey.jersey_vote_tracker import JerseyVoteTracker


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
JERSEY_DIR = PROJECT_ROOT / "jersey_crops"


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("[INFO] Jersey crops directory:", JERSEY_DIR)

    if not JERSEY_DIR.exists():
        print("[ERROR] jersey_crops folder does not exist")
        return

    reader = SmolVLMJerseyReader()

    # 🔒 Temporal validation + locking
    vote_tracker = JerseyVoteTracker(
        min_votes=3,
        min_confidence=0.6
    )

    final_results = {}

    for sid_dir in sorted(JERSEY_DIR.glob("SID_*")):
        sid = sid_dir.name.replace("SID_", "")
        print(f"\nProcessing SID {sid}")

        images = sorted(sid_dir.glob("*.jpg"))
        print(f"Found {len(images)} images for SID {sid}")

        if not images:
            print("⚠️ No images found, skipping")
            continue

        for img_path in images:
            try:
                number, raw = reader.read_jersey_number(img_path)
                print(f"  {img_path.name} → {raw}")

                vote_tracker.add_vote(sid, number)

                # 🔒 Stop once jersey is confidently locked
                if vote_tracker.is_locked(sid):
                    locked = vote_tracker.get_locked(sid)
                    print(
                        f"🔒 SID {sid} LOCKED → Jersey {locked['jersey']} "
                        f"(confidence {locked['confidence']})"
                    )
                    final_results[sid] = locked
                    break

            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")

        if sid not in final_results:
            print(f"⚠️ Jersey not locked for SID {sid}")

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------
    print("\n================ FINAL RESULTS ================\n")

    if not final_results:
        print("No jersey numbers locked.")
    else:
        for sid, info in final_results.items():
            print(
                f"SID {sid} → Jersey {info['jersey']} | "
                f"Confidence {info['confidence']} | "
                f"Votes {info['votes']}"
            )


if __name__ == "__main__":
    main()
