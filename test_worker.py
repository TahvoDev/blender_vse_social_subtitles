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
        rm_punct=False, lang="en",
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
        rm_punct=False, lang="en",
    )
    assert len(subs) == 1, subs
    assert subs[0]["frame_end"] == 35, subs[0]  # clamped from 40 to clip end


def test_words_to_subtitles_rm_punct_keeps_case():
    # Text case is owned by the add-on now; the worker only strips punctuation,
    # and it must do so AFTER using punctuation for break decisions.
    tw = _load("transcribe_worker")
    words = [{"word": "Hello,", "start": 0.0, "end": 0.3},
             {"word": "World!", "start": 0.3, "end": 0.6}]
    subs = tw._words_to_subtitles(
        words, clip_frame_start=0, clip_frame_end=1000,
        fps=25, trim_start_s=0.0,
        max_chars=42, max_lines=2, max_gap_s=1.0,
        rm_punct=True, lang="en",
    )
    assert subs[0]["text"] == "Hello World", subs[0]


def test_word_preservation_and_monotonic_frames():
    # Regression for the old textwrap(...)[:max_lines] truncation: every word
    # must survive, and frames must stay monotonic and non-overlapping.
    tw = _load("transcribe_worker")
    text = "the quick brown fox jumps over the lazy dog again and again"
    words = [{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.28}
             for i, w in enumerate(text.split())]
    subs = tw._words_to_subtitles(
        words, clip_frame_start=0, clip_frame_end=100000,
        fps=30, trim_start_s=0.0,
        max_chars=10, max_lines=2, max_gap_s=1.5,
        rm_punct=False, lang="en",
    )
    got = [t for s in subs for t in s["text"].replace("\n", " ").split()]
    assert got == [w["word"] for w in words], got
    for a, b in zip(subs, subs[1:]):
        assert a["frame_start"] <= a["frame_end"] <= b["frame_start"], (a, b)


def test_terminal_and_clause_guards():
    tw = _load("transcribe_worker")
    assert tw._is_terminal("Hello!")
    assert tw._is_terminal("Done.")
    assert tw._is_terminal("结束。")
    assert not tw._is_terminal("3.5")        # decimal, ends in a digit anyway
    assert not tw._is_terminal("Dr.")        # abbreviation
    assert not tw._is_terminal("U.S.")       # dotted initials
    assert not tw._is_terminal("wait...")    # ellipsis is a clause, not terminal
    assert not tw._is_terminal("word")
    assert tw._is_clause("well,")
    assert tw._is_clause("wait…")
    assert not tw._is_clause("word")


def test_function_word_seam_rebalanced():
    # "the" stranded at the end of a caption should be pulled across to the
    # next (or the blocks merged) so no caption dangles on a function word.
    tw = _load("transcribe_worker")
    words = [
        {"word": "We",    "start": 0.0, "end": 0.3},
        {"word": "drove", "start": 0.3, "end": 0.7},
        {"word": "to",    "start": 0.7, "end": 0.9},
        {"word": "the",   "start": 0.9, "end": 1.1},
        {"word": "park",  "start": 1.1, "end": 1.4},
        {"word": "yesterday.", "start": 1.4, "end": 1.9},
    ]
    subs = tw._words_to_subtitles(
        words, clip_frame_start=0, clip_frame_end=100000,
        fps=30, trim_start_s=0.0,
        max_chars=12, max_lines=2, max_gap_s=1.5,
        rm_punct=False, lang="en",
    )
    for s in subs[:-1]:
        last = s["text"].replace("\n", " ").split()[-1].strip(".,!?;:'\"").lower()
        assert last not in tw._FUNCTION_WORDS["en"], s
    got = [t for s in subs for t in s["text"].replace("\n", " ").split()]
    assert got == [w["word"] for w in words], got


def test_small_block_merged_when_close():
    # A one-word flash caption followed by a close neighbour is absorbed.
    tw = _load("transcribe_worker")
    words = [
        {"word": "Yes.",  "start": 0.0, "end": 0.3},
        {"word": "We",    "start": 0.35, "end": 0.6},
        {"word": "can",   "start": 0.6, "end": 0.8},
        {"word": "start", "start": 0.8, "end": 1.1},
        {"word": "now.",  "start": 1.1, "end": 1.4},
    ]
    subs = tw._words_to_subtitles(
        words, clip_frame_start=0, clip_frame_end=100000,
        fps=30, trim_start_s=0.0,
        max_chars=42, max_lines=2, max_gap_s=1.5,
        rm_punct=False, lang="en",
    )
    got = [t for s in subs for t in s["text"].replace("\n", " ").split()]
    assert got == [w["word"] for w in words], got
    # "Yes." must not stand alone as its own flash caption
    assert not any(s["text"].strip() == "Yes." for s in subs), subs
    # boundary pass must be idempotent (fixpoint reached within the cap)
    ctx = tw._Ctx(lang="en", max_chars=42, max_lines=2, max_dur=6.0, max_cps=17.0,
                  min_chars=12, min_dur=0.8, merge_gap_max=0.45, use_punct=True)
    seed = [[{"word": "a", "start": 0.0, "end": 0.4, "seg_id": 0},
             {"word": "the", "start": 0.4, "end": 0.8, "seg_id": 0}]]
    assert tw._boundary_pass(tw._boundary_pass(seed, ctx), ctx) == seed


def test_duplicate_flagging():
    tw = _load("transcribe_worker")
    dup = [{"word": "the", "start": 0.0, "end": 0.2},
           {"word": "the", "start": 0.2, "end": 0.4},
           {"word": "cat", "start": 0.4, "end": 0.7}]
    assert tw._has_duplicate(dup)
    allowed = [{"word": "that", "start": 0.0, "end": 0.2},
               {"word": "that", "start": 0.2, "end": 0.4}]
    assert not tw._has_duplicate(allowed)


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
