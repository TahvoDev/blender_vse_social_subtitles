"""
align_wav2vec2.py – Word-level forced aligner using torchaudio CTC.

Uses torchaudio.functional.forced_align (available since torchaudio 2.1).
No whisperx, no pyannote, no av, no pandas, no nltk required.
torch._dynamo disabled before any import.

Public API
----------
  load_audio_segment(path, offset_s, duration_s) -> np.ndarray  (16kHz f32 mono)
  align(rel_segments, audio, language, device)   -> [{word,start,end}] or None
    rel_segments : objects with .start/.end/.text/.words (CLIP-RELATIVE seconds)
    audio        : float32 16kHz mono numpy array, clip-relative (t=0 = clip start)
    returns      : [{word,start,end}] clip-relative seconds, or None on failure
"""

import os, sys, subprocess
import numpy as np

os.environ["TORCHDYNAMO_DISABLE"]  = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ---------------------------------------------------------------------------
# Audio loading via ffmpeg — no av dependency
# ---------------------------------------------------------------------------

def load_audio_segment(audio_path: str, offset_s: float,
                        duration_s: float) -> np.ndarray:
    """Decode a clip segment to 16kHz mono float32 via ffmpeg stdout."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(offset_s),
        "-t",  str(min(duration_s, 7200.0)),
        "-i",  audio_path,
        "-ar", "16000", "-ac", "1",
        "-f",  "f32le", "-",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(
            f"ffmpeg failed: {r.stderr.decode(errors='replace')[:300]}"
        )
    audio = np.frombuffer(r.stdout, dtype=np.float32).copy()
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio /= peak
    return audio

# ---------------------------------------------------------------------------
# Language → model mapping
# ---------------------------------------------------------------------------

# torchaudio pipeline names (no HF download, ship with torchaudio)
# Only include ones confirmed present in torchaudio 2.x
_TA_MODELS = {
    "en": "WAV2VEC2_ASR_BASE_960H",
}

# HuggingFace models for other languages (downloaded once to ~/.cache/huggingface)
_HF_MODELS = {
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "ro": "gigant/romanian-wav2vec2",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(language: str, device: str):
    """
    Returns (model, align_dict, pipeline_type) where:
      align_dict    : {char_lower: token_id}
      pipeline_type : 'torchaudio' | 'huggingface'
    Both model types output raw logits — caller applies log_softmax.
    """
    lang = (language or "en").lower().split("-")[0]

    # Try torchaudio bundle first (English only — others removed in 2.x)
    if lang in _TA_MODELS:
        try:
            import torchaudio
            bundle_name = _TA_MODELS[lang]
            bundle = getattr(torchaudio.pipelines, bundle_name)
            model = bundle.get_model().to(device)
            labels = list(bundle.get_labels())
            # torchaudio labels are uppercase; align_dict uses lowercase
            align_dict = {c.lower(): i for i, c in enumerate(labels)}
            return model, align_dict, "torchaudio"
        except Exception as e:
            print(f"[aligner] torchaudio bundle failed: {e}", flush=True)

    # HuggingFace
    hf_name = _HF_MODELS.get(lang, "facebook/wav2vec2-large-960h-lv60-self")
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(hf_name)
    model     = Wav2Vec2ForCTC.from_pretrained(hf_name).to(device)
    align_dict = {
        char.lower(): code
        for char, code in processor.tokenizer.get_vocab().items()
    }
    return model, align_dict, "huggingface"

# ---------------------------------------------------------------------------
# Flatten segments to word list
# ---------------------------------------------------------------------------

def _flatten_words(segments) -> list:
    words = []
    for seg in segments:
        ws = seg.words or []
        if ws:
            for w in ws:
                t = w.word.strip()
                if t:
                    words.append({"word": t,
                                  "start": float(w.start),
                                  "end":   float(w.end)})
        else:
            t = seg.text.strip()
            if t:
                words.append({"word": t,
                              "start": float(seg.start),
                              "end":   float(seg.end)})
    return words

# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------

def align(rel_segments, audio: np.ndarray, language: str, device: str):
    """
    Refine word timestamps via CTC forced alignment.

    rel_segments : segment objects with CLIP-RELATIVE .start/.end timestamps
    audio        : float32 16kHz mono numpy array, clip-relative
    language     : ISO 639-1 language code
    device       : 'cuda' | 'cpu'

    Returns [{word, start, end}] (clip-relative seconds) or None on failure.
    """
    import torch
    import torchaudio.functional as TAF

    words = _flatten_words(rel_segments)
    if not words:
        return None

    try:
        model, align_dict, ptype = _load_model(language, device)
    except Exception as e:
        print(f"[aligner] model load failed: {e}", flush=True)
        return None

    # blank token: '[pad]' or '<pad>' if present, else 0
    blank = align_dict.get("[pad]", align_dict.get("<pad>", 0))
    # space token: '|' for torchaudio, ' ' for some HF models
    space = align_dict.get("|", align_dict.get(" ", None))

    # Get emissions — both torchaudio and HF return raw logits here;
    # torchaudio bundle .get_model() returns a Wav2Vec2Model, not a pipeline,
    # so output is raw features that need log_softmax.
    try:
        wav = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, T)
        with torch.inference_mode():
            out = model(wav)
            # torchaudio model: returns (Tensor, lengths) tuple
            # HF Wav2Vec2ForCTC: returns CausalLMOutput with .logits
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out.logits
        emissions = torch.log_softmax(logits, dim=-1)  # (1, T', V)
    except Exception as e:
        print(f"[aligner] inference failed: {e}", flush=True)
        return None

    # Build token sequence from transcript
    # Space handling: insert space token between words
    transcript = " ".join(w["word"] for w in words).lower()
    tokens = []
    for ch in transcript:
        if ch == " ":
            if space is not None:
                tokens.append(space)
            # if no space token, skip — CTC blank bridges the gap
        elif ch in align_dict:
            tokens.append(align_dict[ch])
        # unknown chars: skip (blank in CTC handles unrecognised phonemes)

    if not tokens:
        print("[aligner] empty token sequence — no recognised chars", flush=True)
        return None

    try:
        targets      = torch.tensor([tokens], dtype=torch.int32, device=device)
        aligned_toks, scores = TAF.forced_align(emissions, targets, blank=blank)
        spans        = TAF.merge_tokens(aligned_toks[0], scores[0])
    except Exception as e:
        print(f"[aligner] forced_align failed: {e}", flush=True)
        return None

    # frames_per_sec: wav2vec2 has a fixed stride of 320 samples at 16kHz = 50fps
    fps = 16000 / 320  # = 50.0

    # Split spans into per-word groups using space as delimiter
    groups, cur = [], []
    for sp in spans:
        if space is not None and sp.token == space:
            if cur:
                groups.append(cur)
                cur = []
        else:
            cur.append(sp)
    if cur:
        groups.append(cur)

    # Map groups back to words
    # If group count doesn't match word count (unknown chars caused token drops),
    # fall back to faster-whisper timestamps for mismatched words.
    result = []
    for i, wd in enumerate(words):
        if i < len(groups) and groups[i]:
            t_start = groups[i][0].start  / fps
            t_end   = groups[i][-1].end   / fps
        else:
            t_start, t_end = wd["start"], wd["end"]
        result.append({"word": wd["word"], "start": t_start, "end": t_end})

    return result or None
