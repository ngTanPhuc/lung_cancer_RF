import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path

def setup_environment():
    """
    Create a virtual environment and install dependencies from requirements.txt.
    Command:
        python run.py setup
    """

    project_dir = Path(__file__).resolve().parent
    venv_dir = project_dir / "venv"
    req_file = project_dir / "requirements.txt"

    if not req_file.exists():
        print(f"requirements.txt not found at: {req_file}")
        return

    # 1. Create venv if not exists
    if not venv_dir.exists():
        print("Creating virtual environment (venv)...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    else:
        print("venv already exists, skipping creation.")

    # 2. Determine pip path
    if os.name == "nt":
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        pip_path = venv_dir / "bin" / "pip"

    if not pip_path.exists():
        print(f"pip not found in venv at: {pip_path}")
        return

    # 3. Install requirements
    print("Installing requirements (this may take several minutes)...")
    install_cmd = [str(pip_path), "--default-timeout=300", "install", "-r", str(req_file)]
    max_retries = 3

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries} ...")
            subprocess.check_call(install_cmd)
            print("✓ Installation completed successfully!")
            break
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt} failed with error code {e.returncode}.")
            if attempt == max_retries:
                print("All attempts failed. Check your internet connection or try again later.")
        except Exception as e:
            print(f"Attempt {attempt} failed:", e)
            if attempt == max_retries:
                print("All attempts failed. Check your internet connection or try again later.")

    # 4. Activation instructions
    print("\nTo activate your venv:")
    if os.name == "nt":
        print("  PowerShell:   .\\venv\\Scripts\\Activate")
        print("  CMD:          venv\\Scripts\\activate")
    else:
        print("  Linux/WSL:    source venv/bin/activate")


def remove_environment():
    """
    Remove the virtual environment folder completely.
    Command:
        python run.py remove
    """
    project_dir = Path(__file__).resolve().parent
    venv_dir = project_dir / "venv"

    if not venv_dir.exists():
        print("No venv directory found — nothing to remove.")
        return

    print(f"Removing virtual environment at: {venv_dir}")
    try:
        shutil.rmtree(venv_dir)
        print("✓ venv removed successfully.")
    except Exception as e:
        print(f"Failed to remove venv: {e}")
        print("Deactive your venv by command: deactivate")


def main():
    parser = argparse.ArgumentParser(description="Project setup runner")
    parser.add_argument("command", help="setup | remove")

    args = parser.parse_args()

    if args.command == "setup":
        setup_environment()
    elif args.command == "remove":
        remove_environment()
    else:
        print(f"Unknown command: {args.command}")
        print("Usage: python run.py [setup | remove]")


if __name__ == "__main__":
    main()