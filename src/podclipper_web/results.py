"""Finished website exports and their advisory quality verdicts."""

from pathlib import Path
import fcntl
import re
import subprocess
import tempfile


def finished_reel_path(message, directory):
    match = re.match(r"^\[\d+/\d+\] Done \((?:publish|review|skip)\): (.+)$", message)
    if match:
        path = Path(match[1]).resolve()
        if path.is_relative_to(directory.resolve()) and path.suffix == ".mp4":
            return path
    return None


def reel_quality(path):
    try:
        sidecar = Path(path).with_suffix(".txt").read_text()
    except OSError:
        return None
    if not sidecar.endswith("\n") or not re.search(r'^Feedback:.*"\s*$', sidecar, re.MULTILINE):
        return None
    match = re.search(r"^Overall:.*→\s*(PUBLISH|REVIEW|SKIP)\s*$", sidecar, re.MULTILINE)
    return match[1].lower() if match else None


def finished_reels(directory, progress):
    completed = list(progress.get("completed", []))
    # Scorecards are appended after rendering, captions, audio and evaluation.
    # This recovers results from older render processes without restarting them.
    for sidecar in sorted((directory / "reels").rglob("*.txt")):
        path = sidecar.with_suffix(".mp4").resolve()
        if (path.is_relative_to(directory.resolve()) and path.is_file()
                and path.with_suffix(".captions.json").is_file()
                and reel_quality(path) is not None and str(path) not in completed):
            completed.append(str(path))
    return completed


def reel_poster(path):
    """Extract one small frame once, including for reels exported before posters existed."""
    path = Path(path)
    poster = path.with_suffix(".poster.jpg")
    if poster.is_symlink():
        raise ValueError("Invalid poster path")
    with poster.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if poster.is_file() and poster.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            return poster
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".poster-", suffix=".jpg", delete=False) as output:
            temporary = Path(output.name)
        try:
            subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                            "-threads", "1", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                            "-vf", "scale=540:-2", "-filter_threads", "1", "-threads", "1",
                            "-q:v", "3", str(temporary)], capture_output=True, timeout=15, check=True)
            temporary.replace(poster)
        finally:
            temporary.unlink(missing_ok=True)
    return poster
