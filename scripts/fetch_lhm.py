"""Download LibreHardwareMonitor (the .NET Framework build) into third_party/, pinned by version and SHA256.

The files are not committed (third_party/ is git-ignored). Run once:

    python scripts/fetch_lhm.py

Reading the CPU temperature additionally needs the PawnIO driver (one-time, administrator):

    winget install --id namazso.PawnIO --exact
"""
import hashlib
import io
import os
import sys
import urllib.request
import zipfile

VERSION = "0.9.6"
URL = f"https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases/download/v{VERSION}/LibreHardwareMonitor.zip"
SHA256 = "086d9f1b5a99e643edc2cfaaac16051685b551e4c5ac0b32a57c58c0e529c001"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(PROJECT_ROOT, "third_party", f"LibreHardwareMonitor-{VERSION}")


def main():
    if os.path.exists(os.path.join(TARGET, "LibreHardwareMonitorLib.dll")):
        print(f"Already present: {TARGET}")
        return 0
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        print(f"SHA256 mismatch: expected {SHA256}, got {digest}. Nothing was extracted.", file=sys.stderr)
        return 1
    os.makedirs(TARGET, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(TARGET)
    print(f"Extracted LibreHardwareMonitor {VERSION} to {TARGET} (sha256 verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
