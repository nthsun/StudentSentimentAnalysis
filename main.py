
from pathlib import Path
import argparse
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent

COMMANDS = {
    "download": "src/data/download_uit_vsfc.py",
    "eda": "src/analysis/explore_uit_vsfc.py",
    "preprocess": "src/analysis/preprocess_uit_vsfc.py",
    "segment": "src/data/segment_uit_vsfc.py",
    "split": "src/data/split_uit_vsfc.py",
    "train": "src/models/train_uit_vsfc_models.py",
    "train-pipeline": "src/models/train_uit_vsfc_pipeline.py",
    "predict": "src/prediction/predict_sentiment.py",
}

def show_commands():
    print("\nUIT-VSFC - STUDENT SENTIMENT ANALYSIS")
    print("=" * 50)
    for name, path in COMMANDS.items():
        status = "OK" if (BASE_DIR / path).is_file() else "MISSING"
        print(f"{name:<18} {status:<10} {path}")
    print("\nUsage: python main.py <command>")

def run_command(command):
    if command not in COMMANDS:
        print(f"Command không hợp lệ: {command}")
        show_commands()
        return 1

    script = BASE_DIR / COMMANDS[command]

    if not script.is_file():
        print(f"Không tìm thấy script: {script}")
        return 1

    print(f"\nĐang chạy chức năng: {command}")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR,
        check=False
    ).returncode

def main():
    parser = argparse.ArgumentParser(
        description="UIT-VSFC Vietnamese Sentiment Analysis"
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Chức năng cần chạy"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liệt kê các chức năng"
    )

    args = parser.parse_args()

    if args.list or not args.command:
        show_commands()
        return 0

    return run_command(args.command)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình.")
        sys.exit(130)
