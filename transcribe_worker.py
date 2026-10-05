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

import os, sys, json, string, textwrap, traceback as _tb

os.environ["TORCHDYNAMO_DISABLE"]  = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# The venv already has everything installed; make the add-on dir importable
# so the worker can find its siblings (align_wav2vec2).
_addon_dir = os.path.dirname(os.path.abspath(__file__))
if _addon_dir not in sys.path:
    sys.path.insert(0, _addon_dir)

# Disable dynamo if torch is already imported.
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
# Aligner (installed; graceful fallback if it errors at runtime)
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
# Audio decoding
# ---------------------------------------------------------------------------

def _decode_audio(path, sampling_rate=16000):
    """
    Decode an audio file to float32 mono at sampling_rate using PyAV.

    Implemented locally rather than via ``faster_whisper.audio.decode_audio``:
    that helper passes ``metadata_errors=...`` to ``av.open``, which PyAV >= 19
    removed, so it raises TypeError there.
    """
    import av
    import numpy as np
    from av.audio.resampler import AudioResampler

    resampler = AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks = []
    with av.open(path, mode="r") as container:
        if not container.streams.audio:
            raise RuntimeError(f"no audio stream in {path}")
        for frame in container.decode(container.streams.audio[0]):
            frame.pts = None
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks, axis=1).reshape(-1)
    return audio.astype(np.float32) / 32768.0


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

    # Decode the clip to 16kHz mono float32 with PyAV. Timestamps are then
    # relative to t=0 of the clip, so no offset math and no system ffmpeg.
    full  = _decode_audio(audio_path, sampling_rate=16000)
    start = max(0, int(trim_start_s * 16000))
    stop  = start + int(trim_duration_s * 16000) if trim_duration_s else None
    clip_audio = full[start:stop]

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
    if _ALIGN_AVAILABLE and segments and clip_audio is not None:
        try:
            aligned = _aligner.align(
                segments, clip_audio,
                language=detected_lang or language or "en",
                device=device,
            )
            if aligned:
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

    # Fallback: faster-whisper word timestamps (clip-relative → file-relative)
    if not all_words:
        all_words = []
        for seg in segments:
            if seg.words:
                for w in seg.words:
                    all_words.append({"word": w.word,
                                      "start": w.start + trim_start_s,
                                      "end":   w.end   + trim_start_s})
            else:
                all_words.append({"word": seg.text.strip(),
                                  "start": seg.start + trim_start_s,
                                  "end":   seg.end   + trim_start_s})
        progress((clip_idx + 0.8) / n_clips,
                 f"Clip {clip_idx+1}: {len(all_words)} words (faster-whisper timestamps)")

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

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        error(f"faster-whisper not found: {e}. Click 'Set up dependencies' in the N-panel.")

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
