"""Run modal training with proper encoding handling."""
import subprocess
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

result = subprocess.run(
    [sys.executable, "-m", "modal", "run", "run_training.py"],
    cwd=r"C:\Develop\workspace\12.JAVIS",
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)

# Write to file to avoid encoding issues
with open("modal_output.log", "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== Return code: {result.returncode} ===\n")

# Print ASCII-safe version
print(f"Return code: {result.returncode}")
print("Output written to modal_output.log")
