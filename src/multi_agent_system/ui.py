import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit web application."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    app_path = root_dir / "app.py"

    if not app_path.exists():
        print(f"Error: app.py not found at {app_path}")
        sys.exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
