"""Append-only result files. Rows are written in one call and existing files are never rewritten."""
import contextlib
import csv
import io
import json
import math
import os
import signal


class SchemaMismatchError(RuntimeError):
    """The CSV on disk has a different header than the rows about to be appended."""


@contextlib.contextmanager
def defer_interrupt():
    """Hold back Ctrl+C until the block has finished.

    Used around the result-file appends of a unit and its campaign-state update, so these
    always happen together or not at all. A Ctrl+C received meanwhile is raised on exit.
    """
    received = []
    try:
        previous = signal.signal(signal.SIGINT, lambda signum, frame: received.append(signum))
    except ValueError:  # Not the main thread: signals cannot be redirected, nothing to defer
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if received:
        raise KeyboardInterrupt


def _format(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return repr(round(float(value), 6))
    return value


def append_rows(csv_path, fieldnames, rows):
    """Append complete rows to a CSV in a single write.

    The whole block is encoded first and written with one call followed by fsync, so an
    interrupt cannot leave a partial line. A header mismatch raises instead of migrating
    the file: results on disk are never modified in place.
    """
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(fieldnames)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if exists:
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != fieldnames:
            raise SchemaMismatchError(
                f"{csv_path} has a different header than the current schema; refusing to append. "
                f"Move the old file aside (it will not be rewritten).")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    if not exists:
        writer.writeheader()
    for row in rows:
        missing = [k for k in fieldnames if k not in row]
        if missing:
            raise KeyError(f"Row for {os.path.basename(csv_path)} is missing columns: {missing}")
        writer.writerow({k: _format(row[k]) for k in fieldnames})

    with open(csv_path, "ab") as f:
        f.write(buffer.getvalue().encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def _json_safe(value):
    """Strict JSON has no NaN/Infinity: store them as null."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json_atomic(path, payload):
    """Write JSON to a temporary file and rename it over the target."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
