"""Consistent console output for every command: stage banners, status marks, tables, timing.

Plain ASCII-safe characters are used where Windows consoles may lack fonts, and output is flushed
immediately so progress is visible in logs written with `python -u`.
"""
import sys
import time

WIDTH = 78
OK, FAIL, TODO, RUN = "[done]", "[FAIL]", "[ -- ]", "[ .. ]"


def _print(text=""):
    print(text, flush=True)


def banner(title, subtitle=None):
    """Large heading for the start of a command or a training run."""
    _print("=" * WIDTH)
    _print(f"  {title}")
    if subtitle:
        _print(f"  {subtitle}")
    _print("=" * WIDTH)


def section(title):
    _print(f"\n--- {title} " + "-" * max(0, WIDTH - len(title) - 5))


def kv(pairs, indent=2):
    """Aligned 'key : value' lines (e.g. a run configuration)."""
    width = max(len(k) for k, _ in pairs)
    for k, v in pairs:
        _print(" " * indent + f"{k:<{width}} : {v}")


def table(headers, rows, indent=2, left=1):
    """Aligned plain-text table; the first `left` columns are left-aligned, the rest right-aligned."""
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows])))
    widths = [max(len(x) for x in c) for c in cols]

    def line(cells):
        return " " * indent + "  ".join(
            (str(c).ljust(w) if i < left else str(c).rjust(w)) for i, (c, w) in enumerate(zip(cells, widths)))
    _print(line(headers))
    _print(" " * indent + "  ".join("-" * w for w in widths))
    for r in rows:
        _print(line(r))


def fmt_seconds(s):
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Stage:
    """Context manager that prints '[i/n] name ... [done] (time)' or '[FAIL]' with the error."""

    def __init__(self, index, total, name):
        self.label = f"[{index}/{total}] {name}"

    def __enter__(self):
        self.t0 = time.time()
        _print(f"\n{RUN} {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        took = fmt_seconds(time.time() - self.t0)
        if exc_type is None:
            _print(f"{OK} {self.label}  ({took})")
        else:
            _print(f"{FAIL} {self.label}  ({took}): {exc}")
        return False


def epoch_line(name, epoch, epochs, loss, val_auc, best_auc, best_epoch, seconds, elapsed, improved):
    """One line per epoch with progress, metrics, best-so-far marker and ETA."""
    remaining = (epochs - epoch) * (elapsed / epoch)
    mark = "  * new best" if improved else ""
    _print(f"  [{name}] epoch {epoch:3d}/{epochs}  loss {loss:.4f}  val AUC {val_auc:.4f}  "
           f"best {best_auc:.4f} @ {best_epoch:<3d} {seconds:4.0f}s/ep  ETA {fmt_seconds(remaining)}{mark}")


def fail(message):
    _print(f"{FAIL} {message}")
    sys.exit(1)
