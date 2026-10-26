import re
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3] / "src" / "mnist" / "monitoring"
# Calls that move data off the device or force a sync. Only transfer.py may use them.
FORBIDDEN = re.compile(r"\.(cpu|numpy|item|tolist)\(")


def test_only_transfer_py_moves_data_off_the_device():
    offenders: list[str] = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "transfer.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if FORBIDDEN.search(code):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == []


def test_the_scan_looks_at_the_real_package():
    assert (PACKAGE / "transfer.py").is_file()
    assert len(list(PACKAGE.glob("*.py"))) >= 6
