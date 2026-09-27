"""
transcribe_worker.py – subprocess worker for Auto-Subtitles VSE add-on.

Pipeline:
  1. ffmpeg        → extract clip audio to 16kHz mono WAV (clean timestamps)
  2. faster-whisper → transcript + word timestamps (clip-relative from t=0)
  3. align_wav2vec2 → wav2vec2 CTC forced alignment for precise timestamps
                      (optional — falls back to faster-whisper timestamps)

torch._dynamo disabled before all imports to prevent Blender subprocess crashes.
"""

import os, sys, site, json, string, textwrap, tempfile, subprocess as _sp, traceback as _tb

os.environ["TORCHDYNAMO_DISABLE"]  = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Patch sys.path for Blender's stripped environment
_addon_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_addon_dir,
           site.getusersitepackages(),
           "/usr/local/lib/python3.14/site-packages",
           "/usr/local/lib64/python3.14/site-packages"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Disable dynamo if torch already loaded
try:
    import torch as _t
    if hasattr(_t, "_dynamo"):
        _t._dynamo.disable()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Progress / error
# ---------------------------------------------------------------------------

def progress(value: float, msg: str = ""):
    print(f"PROGRESS:{value:.4f}:{msg}", flush=True)

def error(msg: str):
    print(f"ERROR:{msg}", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# faster-whisper (required)
# ---------------------------------------------------------------------------
try:
    from faster_whisper import WhisperModel
except ImportError as e:
    error(f"faster-whisper not found: {e}. Install: pip install faster-whisper")

# ---------------------------------------------------------------------------
# Aligner (optional — graceful fallback if torch/torchaudio absent)
# ---------------------------------------------------------------------------
try:
    import align_wav2vec2 as _aligner
    _ALIGN_AVAILABLE = True
except Exception as e:
    _ALIGN_AVAILABLE = False
    print(f"[worker] Aligner unavailable ({e}) – using faster-whisper timestamps", flush=True)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def _apply_case(text, case):
    if case == "upper": return text.upper()
    if case == "lower": return text.lower()
    if case == "title": return text.title()
    return text

# ---------------------------------------------------------------------------
# Word → subtitle blocks
# ---------------------------------------------------------------------------

def _words_to_subtitles(words, clip_frame_start, clip_frame_end,
                         fps, trim_start_s,
                         max_chars, max_lines, max_gap_s,
                         rm_punct, text_case):
    """
    Split words into blocks on char cap OR silence > max_gap_s.
    Each block appears when its first word is spoken.
    Persists until next block starts (seamless) or max_gap_s expires.
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
        # Only clamp if genuinely overlapping (start is before prev word ended)
        if s < prev_end:
            s = prev_end
        e = max(e, s + 0.01)
        clean.append({"word": t, "start": s, "end": e})
        prev_end = e
    if not clean:
        return []
    words = clean

    def to_frame(t):
        return clip_frame_start + int(round((t - trim_start_s) * fps))

    max_total = max_chars * max_lines

    # Pass 1 — group by char cap + silence break
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

    # Pass 2 — raw frame ranges
    subs = []
    for bl in blocks:
        raw = " ".join(w["word"] for w in bl)
        if rm_punct:
            raw = _remove_punctuation(raw)
        raw = _apply_case(raw.strip(), text_case)
        wrapped = "\n".join(textwrap.wrap(raw, width=max_chars)[:max_lines])
        if not wrapped.strip():
            continue
        fs = to_frame(bl[0]["start"])
        fe = to_frame(bl[-1]["end"])
        if fe <= fs:
            fe = fs + 1
        subs.append({"text": wrapped, "frame_start": fs, "frame_end": fe})

    if not subs:
        return []

    # Pass 3 — gap fill / persistence
    max_gap_fr = int(round(max_gap_s * fps))
    for i, s in enumerate(subs):
        nat = s["frame_end"]
        if i + 1 < len(subs):
            gap = subs[i+1]["frame_start"] - nat
            s["frame_end"] = subs[i+1]["frame_start"] if gap <= max_gap_fr \
                             else nat + max_gap_fr
        else:
            s["frame_end"] = nat + max_gap_fr

    # Pass 4 — clamp to clip
    out = []
    for s in subs:
        fs = max(s["frame_start"], clip_frame_start)
        fe = min(s["frame_end"], clip_frame_end) if clip_frame_end else s["frame_end"]
        if fe <= fs:
            fe = fs + 1
        out.append({"text": s["text"], "frame_start": fs, "frame_end": fe})
    return out

# ---------------------------------------------------------------------------
# Transcribe one clip
# ---------------------------------------------------------------------------

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

    # Extract clip audio to a temp WAV at 16kHz mono via ffmpeg.
    # This serves two purposes:
    #   1. Timestamps from faster-whisper are always relative to t=0 of the
    #      WAV, which equals the clip's in-point — no offset math needed.
    #   2. Avoids clip_timestamps API inconsistencies across faster-whisper
    #      versions (the parameter behaves differently in 1.0 vs 1.2).
    tmp_wav = None
    try:
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="as_")
        os.close(fd)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(trim_start_s),
            "-t",  str(trim_duration_s),
            "-i",  audio_path,
            "-ar", "16000", "-ac", "1",
            tmp_wav,
        ]
        r = _sp.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.decode(errors="replace")[:200])
        transcribe_path = tmp_wav
        ts_offset = 0.0   # word timestamps from FW are clip-relative
    except Exception as e:
        print(f"[worker] ffmpeg extraction failed ({e}) – using source file", flush=True)
        transcribe_path = audio_path
        ts_offset = trim_start_s  # must subtract later to get clip-relative

    try:
        seg_gen, info = model.transcribe(
            transcribe_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250, speech_pad_ms=100),
            beam_size=5,
            temperature=0.0,
        )
        segments = list(seg_gen)
        detected_lang = info.language
    except Exception as e:
        # Clean up temp file before re-raising
        if tmp_wav and os.path.exists(tmp_wav):
            try: os.unlink(tmp_wav)
            except: pass
        raise

    # If we fell back to source file, shift timestamps to clip-relative
    if ts_offset > 0:
        class _ShiftedSeg:
            def __init__(self, seg, off):
                self.start = seg.start - off
                self.end   = seg.end   - off
                self.text  = seg.text
                self.words = [_ShiftedWord(w, off) for w in (seg.words or [])]
        class _ShiftedWord:
            def __init__(self, w, off):
                self.word        = w.word
                self.start       = w.start - off
                self.end         = w.end   - off
                self.probability = getattr(w, "probability", 1.0)
        segments = [_ShiftedSeg(s, ts_offset) for s in segments]

    progress((clip_idx + 0.4) / n_clips,
             f"Clip {clip_idx+1}: {len(segments)} segments"
             + (" – aligning…" if _ALIGN_AVAILABLE else " – (no aligner, using FW timestamps)"))

    # At this point segments are clip-relative (t=0 = clip start) because:
    #   - Happy path: transcribed from extracted WAV which starts at t=0
    #   - Fallback path: _ShiftedSeg subtracted ts_offset (= trim_start_s)
    # The aligner expects clip-relative input and returns clip-relative output.
    # _words_to_subtitles expects file-relative input (it subtracts trim_start_s),
    # so we add trim_start_s back when building all_words.

    all_words = None
    if _ALIGN_AVAILABLE and segments:
        try:
            audio = _aligner.load_audio_segment(audio_path, trim_start_s, trim_duration_s)
            aligned = _aligner.align(
                segments, audio,
                language=detected_lang or language or "en",
                device=device,
            )
            if aligned:
                # aligned is clip-relative → add trim_start_s → file-relative
                all_words = [
                    {"word":  w["word"],
                     "start": w["start"] + trim_start_s,
                     "end":   w["end"]   + trim_start_s}
                    for w in aligned
                ]
                progress((clip_idx + 0.8) / n_clips,
                         f"Clip {clip_idx+1}: aligned {len(all_words)} words")
        except Exception as e:
            print(f"[worker] alignment failed: {e}", flush=True)

    # Fallback: faster-whisper word timestamps (already clip-relative)
    if not all_words:
        all_words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    # clip-relative → file-relative for to_frame()
                    all_words.append({"word": w.word,
                                      "start": w.start + trim_start_s,
                                      "end":   w.end   + trim_start_s})
            else:
                all_words.append({"word": seg.text.strip(),
                                  "start": seg.start + trim_start_s,
                                  "end":   seg.end   + trim_start_s})
        progress((clip_idx + 0.8) / n_clips,
                 f"Clip {clip_idx+1}: {len(all_words)} words (faster-whisper timestamps)")

    # Clean up temp WAV
    if tmp_wav and os.path.exists(tmp_wav):
        try:
            os.unlink(tmp_wav)
        except Exception:
            pass

    subs = _words_to_subtitles(
        all_words, frame_start, frame_end, fps, trim_start_s,
        args["max_chars"], args["max_lines"], args.get("max_gap_s", 1.5),
        args["rm_punct"], "default",
    )

    progress((clip_idx + 1) / n_clips,
             f"Clip {clip_idx+1} done – {len(subs)} subtitle block(s)")
    return subs

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    device   = args.get("device", "cpu")

    if not clips:
        error("No clips provided")

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
            progress(0.01, "CUDA unavailable – falling back to CPU …")
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
    for i, clip in enumerate(clips):
        try:
            all_subs.extend(transcribe_clip(clip, model, args, i, len(clips), device))
        except RuntimeError as e:
            msg = str(e)
            if device == "cuda" and any(k in msg for k in
                    ("libcublas","libcuda","cannot be loaded","CUDA")):
                progress(0.01, "CUDA encode failed – retrying on CPU …")
                try:
                    model, device = _load("cpu"), "cpu"
                    all_subs.extend(transcribe_clip(clip, model, args, i, len(clips), device))
                except Exception as e2:
                    error(f"CPU retry failed '{clip['name']}': {e2}\n{_tb.format_exc()}")
            else:
                error(f"Error on clip '{clip['name']}': {e}\n{_tb.format_exc()}")
        except Exception as e:
            error(f"Error on clip '{clip['name']}': {e}\n{_tb.format_exc()}")

    all_subs.sort(key=lambda s: s["frame_start"])

    try:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump({"subtitles": all_subs}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        error(f"Failed to write result: {e}")

    progress(1.0, "Done.")

if __name__ == "__main__":
    main()
