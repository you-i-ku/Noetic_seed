"""minimumtest setup"""
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
VENV = HERE / ".venv"
VENV_PYTHON = VENV / "Scripts" / "python.exe"
VENV_PIP    = VENV / "Scripts" / "pip.exe"
VENV_PW     = VENV / "Scripts" / "playwright.exe"

print("==========================================")
print("  minimumtest setup")
print("==========================================\n")

if not VENV_PYTHON.exists():
    print("Creating virtual environment...")
    r = subprocess.run([sys.executable, "-m", "venv", str(VENV)])
    if r.returncode != 0:
        input("[ERROR] Failed to create venv. Press Enter to exit...")
        sys.exit(1)
else:
    print("Virtual environment already exists.")

print("\nInstalling dependencies (requirements.txt)...")
subprocess.run([str(VENV_PIP), "install", "--quiet", "--upgrade", "pip"])
r = subprocess.run([str(VENV_PIP), "install", "--quiet", "-r", str(HERE / "requirements.txt")])
if r.returncode != 0:
    input("[ERROR] pip install failed. Press Enter to exit...")
    sys.exit(1)

print("\nInstalling Playwright browser (Chromium)...")
r = subprocess.run([str(VENV_PW), "install", "chromium"])
if r.returncode != 0:
    input("[ERROR] Playwright install failed. Press Enter to exit...")
    sys.exit(1)

print("\n==========================================")
print("  Setup complete! Run: run.bat")
print("==========================================")
input("\nPress Enter to exit...")
