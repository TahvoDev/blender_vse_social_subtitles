"""
Self-checks for the Auto-Subtitles worker logic.

Runs with a bare Python interpreter (no third-party deps required); tests that
need numpy/torch are skipped automatically when those are not installed.

    python3 test_worker.py
"""

import importlib.util
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


class SkipTest(Exception):
    pass


def _load(name):
    path = os.path.join(HERE, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_words_to_subtitles_blocks_and_gap():
    tw = _load("transcribe_worker")
    words = [
        {"word": "hello", "start": 0.0, "end": 0.4},
        {"word": "world", "start": 0.4, "end": 0.8},
        {"word": "again", "start": 2.5, "end": 3.0},  # 1.7s silence -> new block
    ]
    subs = tw._words_to_subtitles(
        words, clip_frame_start=100, clip_frame_end=1000,
        fps=10, trim_start_s=0.0,
        max_chars=42, max_lines=2, max_gap_s=1.0,
        rm_punct=False, text_case="default",
    )
    assert len(subs) == 2, subs
    assert subs[0]["text"] == "hello world", subs[0]
    assert subs[0]["frame_start"] == 100, subs[0]
    # gap fill caps at max_gap_s (1.0s = 10 frames) past the natural end (108)
    assert subs[0]["frame_end"] == 118, subs[0]
    assert subs[1]["frame_start"] == 125, subs[1]
    assert subs[1]["frame_end"] == 140, subs[1]


def test_words_to_subtitles_clamp_to_clip():
    tw = _load("transcribe_worker")
    words = [{"word": "tail", "start": 2.5, "end": 3.0}]
    subs = tw._words_to_subtitles(
        words, clip_frame_start=0, clip_frame_end=35,
        fps=10, trim_start_s=0.0,
        max_chars=42, max_lines=2, max_gap_s=1.0,
        rm_punct=False, text_case="default",
    )
    assert len(subs) == 1, subs
    assert subs[0]["frame_end"] == 35, subs[0]  # clamped from 40 to clip end


def test_words_to_subtitles_punctuation_and_case():
    tw = _load("transcribe_worker")
    words = [{"word": "Hello,", "start": 0.0, "end": 0.3},
             {"word": "World!", "start": 0.3, "end": 0.6}]
    subs = tw._words_to_subtitles(
        words, clip_frame_start=0, clip_frame_end=1000,
        fps=25, trim_start_s=0.0,
        max_chars=42, max_lines=2, max_gap_s=1.0,
        rm_punct=True, text_case="upper",
    )
    assert subs[0]["text"] == "HELLO WORLD", subs[0]


def test_forced_alignment_trellis():
    try:
        import numpy as np
        import torch
    except Exception as e:
        raise SkipTest(str(e))

    align = _load("align_wav2vec2")

    # 5 frames, vocab {0:blank, 1:'a', 2:'b'}; 'a' on frames 0-1, 'b' on 3-4.
    def row(hi):
        v = [0.0, 0.0, 0.0]
        for i in range(3):
            v[i] = -0.05 if i == hi else -5.0
        return v

    emission = torch.tensor([row(1), row(1), row(0), row(2), row(2)])
    tokens = [1, 2]
    trellis = align._get_trellis(emission, tokens, 0)
    path = align._backtrack(trellis, emission, tokens, 0)
    assert path is not None, "backtrack returned None"
    spans = align._merge_repeats(path, tokens)
    labels = [s.token for s in spans]
    assert labels == [1, 2], labels


def test_decode_audio_trim_and_normalize():
    try:
        import numpy as np
        import av  # noqa: F401
    except Exception as e:
        raise SkipTest(str(e))

    import wave
    import struct
    import math
    import tempfile

    tw = _load("transcribe_worker")

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(
                struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / 16000)))
                for i in range(16000)
            ))

        full = tw._decode_audio(path, sampling_rate=16000)
        assert full.dtype == np.float32, full.dtype
        assert full.ndim == 1
        assert abs(full.size - 16000) < 1000, full.size
        # peak-normalised to ~1.0 (aligner is amplitude-sensitive)
        assert 0.9 <= float(np.abs(full).max()) <= 1.0001, float(np.abs(full).max())

        clip = tw._decode_audio(path, 0.25, 0.25, sampling_rate=16000)
        assert abs(clip.size - 4000) < 1000, clip.size
    finally:
        os.unlink(path)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = skipped = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except SkipTest as e:
            skipped += 1
            print(f"skip  {t.__name__} ({e})")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    passed = len(tests) - failures - skipped
    print(f"\n{passed} passed, {skipped} skipped, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
