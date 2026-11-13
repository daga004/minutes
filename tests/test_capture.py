import subprocess

from minutes import capture
from minutes import session as sess


def _header_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    return None if out in ("", "N/A") else float(out)


def test_fix_chunk_headers(cfg):
    s = sess.create(cfg)
    # Same muxer settings as live capture: segmenting leaves bad STREAMINFO.
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=5",
                    "-ac", "1", "-ar", "16000", "-c:a", "flac", "-f", "segment", "-segment_time", "2",
                    "-reset_timestamps", "1", str(s.audio_dir("me") / "%05d.flac")], check=True)
    chunks = s.chunks("me")
    assert len(chunks) == 3
    assert capture.fix_chunk_headers(s) == 3
    durs = [_header_duration(c) for c in s.chunks("me")]
    assert all(d is not None for d in durs), durs
    assert abs(sum(durs) - 5.0) < 0.2 and max(durs) <= 2.1, durs
    assert not list(s.audio_dir("me").glob(".*"))  # no temp files left behind
