"""
transcribe_worker.py – subprocess worker for Auto-Subtitles VSE add-on.

Pipeline:
  1. PyAV → decode clip audio to 16kHz mono float32 (no system ffmpeg)
  2. faster-whisper → transcript + word timestamps (clip-relative from t=0)
  3. align_wav2vec2 → wav2vec2 CTC forced alignment (falls back to
                      faster-whisper timestamps if it fails)

Runs in the add-on's private venv, never inside Blender's process.
torch._dynamo disabled before any import to prevent crashes.
"""

import os, sys, json, string, re, traceback as _tb

os.environ["TORCHDYNAMO_DISABLE"]  = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# The venv already has everything installed; make the add-on dir importable
# so the worker can find its siblings (align_wav2vec2).
_addon_dir = os.path.dirname(os.path.abspath(__file__))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

try:
    import torch as _t
    if hasattr(_t, "_dynamo"):
        _t._dynamo.disable()
except Exception:
    pass


def progress(value: float, msg: str = ""):
    print(f"PROGRESS:{value:.4f}:{msg}", flush=True)

def error(msg: str):
    print(f"ERROR:{msg}", flush=True)
    sys.exit(1)

try:
    import align_wav2vec2 as _aligner
    _ALIGN_AVAILABLE = True
except Exception as e:
    _ALIGN_AVAILABLE = False
    print(f"[worker] Aligner unavailable ({e}) – using faster-whisper timestamps", flush=True)


def _remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

DEFAULT_MIN_CHARS = 12
DEFAULT_MIN_DUR = 0.8
DEFAULT_MERGE_GAP_MAX = 0.45
DEFAULT_MAX_DUR = 6.0
DEFAULT_MAX_CPS = 17.0
_PASS_CAP = 4
_PUNCT_DENSITY_MIN = 0.02

_CJK_LANGS = {"zh", "ja", "ko"}

# Function words (articles, prepositions, conjunctions, auxiliaries, possessive
# determiners) whose presence at a caption edge reads as a broken phrase.
# "Absent language = rule disabled" — never falls back to English, so e.g.
# Spanish "de/la/que" are not judged by English rules.
_FUNCTION_WORDS = {
    "en": {
        "the", "a", "an", "and", "or", "but", "nor", "so", "yet", "for",
        "of", "to", "in", "on", "at", "by", "with", "from", "into", "onto",
        "upon", "over", "under", "above", "below", "between", "among",
        "through", "during", "before", "after", "against", "about", "as",
        "than", "that", "which", "who", "whom", "whose", "if", "then",
        "because", "while", "when", "where", "whether", "is", "are", "was",
        "were", "be", "been", "being", "am", "do", "does", "did", "has",
        "have", "had", "will", "would", "shall", "should", "can", "could",
        "may", "might", "must", "not", "no", "its", "his", "her", "their",
        "our", "your", "my", "him", "them", "us", "me", "this", "these",
        "those", "there", "here",
    },
}

# Words that, starting a caption, imply the previous sentence carries on.
_CONTINUATION_WORDS = {
    "en": {
        "and", "but", "or", "so", "then", "because", "which", "that", "who",
        "whom", "whose", "if", "when", "while", "where", "as", "nor", "yet",
        "though", "although", "unless", "until", "since",
    },
}

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "fig", "al", "inc", "ltd", "co", "corp", "dept", "est",
    "approx", "min", "max", "u.s", "u.k", "a.m", "p.m", "cf", "vol",
}

_DUP_ALLOW = {"that that", "had had", "no no", "very very",
              "really really", "ha ha", "bye bye", "yes yes"}


class _Ctx:
    """Tunables for one _words_to_subtitles call."""
    __slots__ = ("lang", "max_chars", "max_lines", "max_dur", "max_cps",
                 "min_chars", "min_dur", "merge_gap_max", "use_punct")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _is_ellipsis(t):
    return t.endswith("…") or t.endswith("...")


def _is_terminal(word):
    t = word.strip().rstrip("\"')]}”’»》」』")
    if not t or _is_ellipsis(t):
        return False
    ch = t[-1]
    if ch in "。！？":
        return True
    if ch in ".!?":
        if ch == ".":
            core = t[:-1].lower()
            if core in _ABBREV or (len(core) == 1 and core.isalpha()):
                return False
            if re.fullmatch(r"([a-z]\.)+[a-z]", core):
                return False
        return True
    return False


_CLAUSE_RE = re.compile(r"[,;:—–、，；]+\s*[\"'\)\]}”’»》」』]*$")


def _is_clause(word):
    t = word.strip()
    if not t:
        return False
    if _is_ellipsis(t):
        return True
    return bool(_CLAUSE_RE.search(t))


def _token(word):
    return word.strip().strip(string.punctuation).lower()


def _in_set(word, table, lang):
    s = table.get(lang)
    if not s:
        return False
    if "'" in word or "’" in word:
        return False
    tok = _token(word)
    return bool(tok) and tok in s


def _is_function_word(word, lang):
    return _in_set(word, _FUNCTION_WORDS, lang)


def _is_continuation_word(word, lang):
    return _in_set(word, _CONTINUATION_WORDS, lang)


def _punct_density(words):
    if not words:
        return 0.0
    n = sum(1 for w in words if _is_terminal(w["word"]) or _is_clause(w["word"]))
    return n / len(words)


def _span(words):
    return words[-1]["end"] - words[0]["start"]


def _chars(words):
    return sum(len(w["word"]) for w in words) + len(words) - 1


def _wrap_block(words, max_chars, max_lines):
    """
    Lay words into <= max_lines lines, each <= max_chars, keeping every word.

    Returns the list of lines, or None if it cannot fit. A single word longer
    than max_chars (W1) is allowed to overflow on its own line — the caller
    flags it. Line breaks are chosen by DP to minimise raggedness rather than a
    fixed-width greedy wrap, so no word is ever truncated.
    """
    toks = [w["word"] for w in words]
    n = len(toks)
    if n == 0:
        return None

    plen = [0] * (n + 1)
    for k in range(n):
        plen[k + 1] = plen[k] + len(toks[k])

    def width(i, j):
        return plen[j] - plen[i] + (j - i - 1)

    INF = float("inf")
    cost = [[INF] * (max_lines + 1) for _ in range(n + 1)]
    nxt = [[-1] * (max_lines + 1) for _ in range(n + 1)]
    for l in range(max_lines + 1):
        cost[n][l] = 0
    orphan = max_chars * max_chars

    for i in range(n - 1, -1, -1):
        for l in range(1, max_lines + 1):
            best, bestj = INF, -1
            for j in range(i + 1, n + 1):
                w = width(i, j)
                single = (j - i == 1)
                if w > max_chars and not single:
                    break
                line_cost = 0 if j == n else (max_chars - w) ** 2
                if single and j != n and w <= max_chars:
                    line_cost += orphan
                if j < n and _is_clause(toks[j - 1]):
                    line_cost = max(0, line_cost - orphan // 2)
                c = line_cost + cost[j][l - 1]
                if c < best:
                    best, bestj = c, j
            cost[i][l], nxt[i][l] = best, bestj

    if cost[0][max_lines] == INF:
        return None
    lines, i, l = [], 0, max_lines
    while i < n:
        j = nxt[i][l]
        if j <= i:
            return None
        lines.append(" ".join(toks[i:j]))
        if j < n:
            l -= 1
            if l < 1:
                return None
        i = j
    return lines


def _structural_fits(words, ctx):
    """Cap/line/duration fit — the parts a split can actually fix."""
    if not words or _span(words) > ctx.max_dur:
        return False
    return _wrap_block(words, ctx.max_chars, ctx.max_lines) is not None


def _fits(words, ctx):
    """Full fit, including reading speed. Used to gate merges/rebreatks so a
    remedy never creates an over-dense block."""
    if not _structural_fits(words, ctx):
        return False
    span = _span(words)
    if span > 0 and _chars(words) / span > ctx.max_cps:
        return False
    return True


def _boundary_badness(prev, next_, ctx):
    """How badly the seam between two blocks reads (0 = fine).

    One combined check for F2 (function word before), F3 (continuation word
    after) and F4 (no terminal punctuation). Only seams closer than
    merge_gap_max are candidates, so a silence the seed deliberately split on
    (gap > max_gap_s) is never merged back.
    """
    if not prev or not next_:
        return 0
    if next_[0]["start"] - prev[-1]["end"] > ctx.merge_gap_max:
        return 0
    bad = 0
    if ctx.use_punct and not _is_terminal(prev[-1]["word"]):
        bad += 1
    if _is_function_word(prev[-1]["word"], ctx.lang):
        bad += 1
    if _is_continuation_word(next_[0]["word"], ctx.lang):
        bad += 1
    return bad


def _rebreak(prev, next_, ctx):
    """Shift words across the seam if that strictly lowers its badness.

    Returns (new_prev, new_next) or None. Both sides must pass _fits.
    """
    cur = _boundary_badness(prev, next_, ctx)
    best = None
    for k in range(1, len(prev)):
        p, nx = prev[:-k], prev[-k:] + next_
        if not (_fits(p, ctx) and _fits(nx, ctx)):
            continue
        nb = _boundary_badness(p, nx, ctx)
        if nb < cur and (best is None or nb < best[0]):
            best = (nb, (p, nx))
    for k in range(1, len(next_)):
        p, nx = prev + next_[:k], next_[k:]
        if not (_fits(p, ctx) and _fits(nx, ctx)):
            continue
        nb = _boundary_badness(p, nx, ctx)
        if nb < cur and (best is None or nb < best[0]):
            best = (nb, (p, nx))
    return best[1] if best else None


def _split_words(words, ctx):
    """Best place to split a block: near the middle, favouring a break after
    terminal punctuation, then a clause mark, then a segment boundary (soft)."""
    if len(words) < 2:
        return None
    best = None
    for k in range(1, len(words)):
        score = abs(k - (len(words) - k))
        if ctx.use_punct and _is_terminal(words[k - 1]["word"]):
            score -= 5
        elif ctx.use_punct and _is_clause(words[k - 1]["word"]):
            score -= 2
        elif words[k].get("seg_id") != words[k - 1].get("seg_id"):
            score -= 1
        if best is None or score < best[0]:
            best = (score, k)
    k = best[1]
    return words[:k], words[k:]


def _fit_blocks(blocks, ctx, depth=0):
    """Split any seed block that overflows the caps before the pass runs."""
    out = []
    for b in blocks:
        if depth < 16 and not _structural_fits(b, ctx):
            sp = _split_words(b, ctx)
            if sp:
                out.extend(_fit_blocks([sp[0], sp[1]], ctx, depth + 1))
                continue
        out.append(b)
    return out


def _boundary_pass(blocks, ctx):
    """Improve bad seams by rebreak/merge, at most _PASS_CAP sweeps.

    A rebreak can re-dirty the previous seam, so this is a bounded sweep, not a
    strict fixpoint; _PASS_CAP is the safety net.
    """
    if len(blocks) < 2:
        return blocks
    for _ in range(_PASS_CAP):
        changed = False
        i = 0
        while i < len(blocks) - 1:
            if _boundary_badness(blocks[i], blocks[i + 1], ctx) > 0:
                cand = _rebreak(blocks[i], blocks[i + 1], ctx)
                if cand is not None:
                    blocks[i], blocks[i + 1] = cand
                    changed = True
                else:
                    merged = blocks[i] + blocks[i + 1]
                    if _fits(merged, ctx):
                        blocks[i:i + 2] = [merged]
                        changed = True
            i += 1
        if not changed:
            break
    return blocks


def _is_small(words, ctx):
    if not words:
        return False
    if ctx.min_chars > 0 and _chars(words) < ctx.min_chars:
        return True
    if ctx.min_dur > 0 and _span(words) < ctx.min_dur:
        return True
    return False


def _small_pass(blocks, ctx):
    """Merge orphan/micro-island blocks (F1) into a neighbour when close."""
    if ctx.min_chars <= 0 and ctx.min_dur <= 0:
        return blocks
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(blocks):
            if i > 0 and _is_small(blocks[i], ctx):
                gap = blocks[i][0]["start"] - blocks[i - 1][-1]["end"]
                if gap <= ctx.merge_gap_max and _fits(blocks[i - 1] + blocks[i], ctx):
                    blocks[i - 1:i + 1] = [blocks[i - 1] + blocks[i]]
                    changed = True
                    continue
            i += 1
        i = 0
        while i < len(blocks) - 1:
            if _is_small(blocks[i], ctx):
                gap = blocks[i + 1][0]["start"] - blocks[i][-1]["end"]
                if gap <= ctx.merge_gap_max and _fits(blocks[i] + blocks[i + 1], ctx):
                    blocks[i:i + 2] = [blocks[i] + blocks[i + 1]]
                    changed = True
                    continue
            i += 1
    return blocks


def _has_duplicate(words):
    """W4: exact duplicate tokens within a tiny gap, honouring an allowlist of
    genuine repetitions ("had had", "no no no")."""
    for a, b in zip(words, words[1:]):
        if b["start"] - a["end"] > 0.2:
            continue
        ka = a["word"].strip(string.punctuation).lower()
        kb = b["word"].strip(string.punctuation).lower()
        if ka and ka == kb and f"{ka} {kb}" not in _DUP_ALLOW:
            return True
    return False


def _words_to_subtitles(words, clip_frame_start, clip_frame_end,
                         fps, trim_start_s,
                         max_chars, max_lines, max_gap_s,
                         rm_punct, lang="en",
                         min_chars=DEFAULT_MIN_CHARS,
                         min_dur=DEFAULT_MIN_DUR,
                         merge_gap_max=DEFAULT_MERGE_GAP_MAX,
                         max_dur=DEFAULT_MAX_DUR,
                         max_cps=DEFAULT_MAX_CPS):
    """
    Split words into blocks, then fix awkward seams/fragments.

    Seeding is the historical char-cap / silence split. On top of it:
      * balanced, non-truncating line wrap (never drops a word),
      * a boundary pass that rebreatks or merges wrongly-cut phrases (F2/F3/F4),
      * a small-block pass that folds orphan/flash captions into a neighbour
        (F1),
      * conservative duplicate flagging (W4).

    Punctuation- and function-word-based decisions read the ORIGINAL tokens;
    rm_punct is applied last so the cues survive. Text case is owned by the
    add-on, not here. Each block appears when its first word is spoken and
    persists until the next block starts (seamless) or max_gap_s expires.
    """
    if not words:
        return []

    # Sanitise: drop empties, fix genuinely overlapping timestamps only
    # Do NOT push a word's start forward just because it equals prev_end —
    # that causes cumulative drift over long clips.
    clean, prev_end = [], -1.0
    for w in words:
        t = w["word"].strip()
        if not t:
            continue
        s = float(w["start"])
        e = float(w["end"])
        if s < prev_end:
            s = prev_end
        e = max(e, s + 0.01)
        clean.append({"word": t, "start": s, "end": e, "seg_id": w.get("seg_id")})
        prev_end = e
    if not clean:
        return []
    words = clean

    def to_frame(t):
        return clip_frame_start + int(round((t - trim_start_s) * fps))

    base_lang = (lang or "en").lower().split("-")[0]
    ctx = _Ctx(
        lang=base_lang,
        max_chars=max_chars, max_lines=max_lines,
        max_dur=max_dur, max_cps=max_cps,
        min_chars=min_chars, min_dur=min_dur, merge_gap_max=merge_gap_max,
        use_punct=(base_lang not in _CJK_LANGS
                   and _punct_density(words) >= _PUNCT_DENSITY_MIN),
    )

    max_total = max_chars * max_lines

    blocks, block = [], []
    for w in words:
        if block:
            gap = w["start"] - block[-1]["end"]
            cur_len = sum(len(b["word"]) for b in block) + len(block) - 1
            if cur_len + 1 + len(w["word"]) > max_total or gap > max_gap_s:
                blocks.append(block)
                block = []
        block.append(w)
    if block:
        blocks.append(block)

    blocks = _fit_blocks(blocks, ctx)
    blocks = _boundary_pass(blocks, ctx)
    blocks = _small_pass(blocks, ctx)

    subs = []
    for bi, bl in enumerate(blocks):
        lines = _wrap_block(bl, max_chars, max_lines)
        if lines is None:
            lines = [" ".join(w["word"] for w in bl)]
        raw = "\n".join(lines)
        if rm_punct:
            raw = _remove_punctuation(raw)
        if not raw.strip():
            continue

        flags = []
        if any(len(ln) > max_chars for ln in lines):
            flags.append("W1")
        if _has_duplicate(bl):
            flags.append("W4")
        if _is_small(bl, ctx):
            flags.append("F1")
        if bi + 1 < len(blocks) and _boundary_badness(bl, blocks[bi + 1], ctx) > 0:
            # A seam the pass could not improve within the caps. F4 (no
            # terminal punctuation mid-sentence) is left out on purpose: it is
            # the normal case, so flagging it would drown the others.
            if _is_function_word(bl[-1]["word"], ctx.lang):
                flags.append("F2")
            if _is_continuation_word(blocks[bi + 1][0]["word"], ctx.lang):
                flags.append("F3")
        span = _span(bl)
        if span > max_dur:
            flags.append("DUR")
        if span > 0 and _chars(bl) / span > max_cps:
            flags.append("CPS")

        fs = to_frame(bl[0]["start"])
        fe = to_frame(bl[-1]["end"])
        if fe <= fs:
            fe = fs + 1
        sub = {"text": raw, "frame_start": fs, "frame_end": fe}
        if flags:
            sub["flags"] = flags
        subs.append(sub)

    if not subs:
        return []

    max_gap_fr = int(round(max_gap_s * fps))
    for i, s in enumerate(subs):
        nat = s["frame_end"]
        if i + 1 < len(subs):
            gap = subs[i+1]["frame_start"] - nat
            s["frame_end"] = subs[i+1]["frame_start"] if gap <= max_gap_fr \
                             else nat + max_gap_fr
        else:
            s["frame_end"] = nat + max_gap_fr

    out = []
    for s in subs:
        fs = max(s["frame_start"], clip_frame_start)
        fe = min(s["frame_end"], clip_frame_end) if clip_frame_end else s["frame_end"]
        if fe <= fs:
            fe = fs + 1
        sub = {"text": s["text"], "frame_start": fs, "frame_end": fe}
        if s.get("flags"):
            sub["flags"] = s["flags"]
        out.append(sub)
    return out


def _iter_resampled(container, stream, resampler):
    """Yield 1-D int16 mono arrays from a PyAV container (incl. flush)."""
    for frame in container.decode(stream):
        frame.pts = None
        for out in resampler.resample(frame):
            yield out.to_ndarray().reshape(-1)
    for out in resampler.resample(None):
        yield out.to_ndarray().reshape(-1)


def _decode_audio(path, start_s=0.0, duration_s=None, sampling_rate=16000):
    """
    Decode a clip to peak-normalised float32 mono at sampling_rate using PyAV.

    Seeks to ``start_s`` and keeps only the clip window, so peak memory is the
    clip, not the whole source file. Output is peak-normalised because the
    wav2vec2 aligner is amplitude-sensitive.

    ponytail: local PyAV decode instead of faster_whisper.audio.decode_audio,
    which hardcodes the ``metadata_errors`` kwarg PyAV >= 19 removed (still
    unguarded in the latest release, 1.2.1). Drop this and delegate once
    faster-whisper guards it AND exposes seek/normalise — but we also want the
    streaming trim, so likely keep it.
    """
    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    start_sample = max(0, int(round(start_s * sampling_rate)))
    want_samples = int(round(duration_s * sampling_rate)) if duration_s else None

    resampler = AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    kept, got, seen = [], 0, 0
    with av.open(path, mode="r") as container:
        if not container.streams.audio:
            raise RuntimeError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        if start_sample:
            # Seek is keyframe-based; the skip loop below trims to exact start.
            try:
                container.seek(int(start_s * av.time_base), backward=True)
            except Exception:
                pass
        for arr in _iter_resampled(container, stream, resampler):
            n = arr.size
            if seen + n <= start_sample:
                seen += n
                continue
            lo = max(0, start_sample - seen)
            hi = n if want_samples is None else min(n, lo + (want_samples - got))
            if hi > lo:
                kept.append(arr[lo:hi])
                got += hi - lo
            seen += n
            if want_samples is not None and got >= want_samples:
                break

    if not kept:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(kept).astype(np.float32)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    audio /= peak if peak > 1e-6 else 32768.0
    return audio


def transcribe_clip(clip, model, args, clip_idx, n_clips, device):
    audio_path      = clip["audio_path"]
    frame_start     = clip["frame_start"]
    frame_end       = clip["frame_end"]
    trim_start_s    = clip.get("trim_start_s", 0.0)
    trim_duration_s = clip.get("trim_duration_s", 7200.0)
    fps             = clip.get("fps", args["fps"])
    language        = args.get("language")

    progress(clip_idx / n_clips,
             f"Transcribing clip {clip_idx+1}/{n_clips}: {clip['name']} …")

    clip_audio = _decode_audio(audio_path, trim_start_s, trim_duration_s,
                               sampling_rate=16000)

    seg_gen, info = model.transcribe(
        clip_audio,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=100),
        beam_size=5,
        temperature=0.0,
    )
    segments = list(seg_gen)
    detected_lang = info.language

    progress((clip_idx + 0.4) / n_clips,
             f"Clip {clip_idx+1}: {len(segments)} segments"
             + (" – aligning…" if _ALIGN_AVAILABLE else " – (no aligner, using FW timestamps)"))

    # Segments are clip-relative here. The aligner takes the clip-relative
    # audio array; _words_to_subtitles wants file-relative times (it subtracts
    # trim_start_s), so add trim_start_s back when building all_words.
    all_words = None
    if _ALIGN_AVAILABLE and segments:
        try:
            aligned = _aligner.align(
                segments, clip_audio,
                language=detected_lang or language or "en",
                device=device,
            )
            if aligned:
                # Map aligned words back to their whisper segment so the
                # segment boundary can act as a soft break hint. The per-segment
                # counts must match align_wav2vec2._flatten_words exactly or
                # every later word shifts.
                seg_ids = []
                for si, seg in enumerate(segments):
                    ws = seg.words or []
                    if ws:
                        n = sum(1 for w in ws if w.word.strip())
                    else:
                        n = 1 if seg.text.strip() else 0
                    seg_ids.extend([si] * n)
                all_words = [
                    {"word":  w["word"],
                     "start": w["start"] + trim_start_s,
                     "end":   w["end"]   + trim_start_s,
                     "seg_id": seg_ids[i] if i < len(seg_ids) else None}
                    for i, w in enumerate(aligned)
                ]
                progress((clip_idx + 0.8) / n_clips,
                         f"Clip {clip_idx+1}: aligned {len(all_words)} words")
        except Exception as e:
            print(f"[worker] alignment failed: {e}", flush=True)

    if not all_words:
        all_words = []
        for si, seg in enumerate(segments):
            if seg.words:
                for w in seg.words:
                    all_words.append({"word": w.word,
                                      "start": w.start + trim_start_s,
                                      "end":   w.end   + trim_start_s,
                                      "seg_id": si})
            else:
                all_words.append({"word": seg.text.strip(),
                                  "start": seg.start + trim_start_s,
                                  "end":   seg.end   + trim_start_s,
                                  "seg_id": si})
        progress((clip_idx + 0.8) / n_clips,
                 f"Clip {clip_idx+1}: {len(all_words)} words (faster-whisper timestamps)")

    subs = _words_to_subtitles(
        all_words, frame_start, frame_end, fps, trim_start_s,
        args["max_chars"], args["max_lines"], args.get("max_gap_s", 1.5),
        args["rm_punct"], lang=detected_lang or language or "en",
        min_chars=args.get("min_chars", DEFAULT_MIN_CHARS),
        min_dur=args.get("min_dur", DEFAULT_MIN_DUR),
        merge_gap_max=args.get("merge_gap_max", DEFAULT_MERGE_GAP_MAX),
    )

    progress((clip_idx + 1) / n_clips,
             f"Clip {clip_idx+1} done – {len(subs)} subtitle block(s)")
    return subs


def main():
    if len(sys.argv) < 3:
        error("Usage: transcribe_worker.py <args_json> <result_json>")

    try:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            args = json.load(f)
    except Exception as e:
        error(f"Failed to read args JSON: {e}")

    clips    = args["clips"]
    model_sz = args.get("model", "base")
    device   = args.get("device", "auto")

    if device == "auto":
        # Prefer the GPU; the model load below falls back to CPU on any CUDA error.
        device = "cuda"

    if not clips:
        error("No clips provided")

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        error(f"faster-whisper not found: {e}. Re-run Generate Subtitles to reinstall.")

    progress(0.0, f"Loading model '{model_sz}' on {device}"
             + (" + wav2vec2 aligner" if _ALIGN_AVAILABLE else " (no aligner)") + " …")

    def _load(dev):
        ct = "float16" if dev == "cuda" else "int8"
        return WhisperModel(model_sz, device=dev, compute_type=ct)

    try:
        model = _load(device)
    except RuntimeError as e:
        msg = str(e)
        if device == "cuda" and any(k in msg for k in
                ("libcublas","libcuda","cannot be loaded","CUDA")):
            reason = msg.splitlines()[0][:120] if msg else "unknown error"
            progress(0.01, f"CUDA unavailable – falling back to CPU ({reason}) …")
            try:
                model, device = _load("cpu"), "cpu"
            except Exception as e2:
                error(f"CPU fallback failed: {e2}\n{_tb.format_exc()}")
        else:
            error(f"Model load failed: {e}\n{_tb.format_exc()}")
    except Exception as e:
        error(f"Model load failed: {e}\n{_tb.format_exc()}")

    progress(0.02, f"Model loaded on {device}.")

    all_subs = []
    failed = []
    for i, clip in enumerate(clips):
        try:
            all_subs.extend(transcribe_clip(clip, model, args, i, len(clips), device))
        except RuntimeError as e:
            msg = str(e)
            if device == "cuda" and any(k in msg for k in
                    ("libcublas","libcuda","cannot be loaded","CUDA")):
                reason = msg.splitlines()[0][:120] if msg else "unknown error"
                progress(0.01, f"CUDA encode failed – retrying on CPU ({reason}) …")
                try:
                    model, device = _load("cpu"), "cpu"
                    all_subs.extend(transcribe_clip(clip, model, args, i, len(clips), device))
                except Exception as e2:
                    failed.append(clip["name"])
                    print(f"[worker] clip '{clip['name']}' failed on CPU retry: {e2}", flush=True)
                    print(_tb.format_exc(), flush=True)
            else:
                failed.append(clip["name"])
                print(f"[worker] clip '{clip['name']}' failed: {e}", flush=True)
                print(_tb.format_exc(), flush=True)
        except Exception as e:
            # One bad file must not sink the rest of the job.
            failed.append(clip["name"])
            print(f"[worker] clip '{clip['name']}' failed: {e}", flush=True)
            print(_tb.format_exc(), flush=True)

    all_subs.sort(key=lambda s: s["frame_start"])

    try:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump({"subtitles": all_subs}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        error(f"Failed to write result: {e}")

    flagged = sum(1 for s in all_subs if s.get("flags"))
    if flagged:
        print(f"[worker] {flagged} subtitle block(s) flagged for review "
              f"(see the 'flags' field in the result)", flush=True)

    if failed:
        progress(1.0, f"Done ({len(failed)}/{len(clips)} clip(s) failed"
                      + (f", {flagged} block(s) flagged" if flagged else "")
                      + f": {', '.join(failed)})")
    else:
        progress(1.0, f"Done ({flagged} block(s) flagged)." if flagged else "Done.")

if __name__ == "__main__":
    main()
