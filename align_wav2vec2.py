"""
align_wav2vec2.py – Word-level forced alignment using a wav2vec2 CTC model.

Pure-PyTorch CTC forced alignment (trellis + backtrack + merge), adapted from
the PyTorch forced-alignment tutorial:
  https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html
and the equivalent implementation in WhisperX (BSD-2-Clause):
  https://github.com/m-bain/whisperX

No torchaudio, whisperx, pyannote, av, pandas, or nltk required.
torch._dynamo is disabled before any torch import.

Public API
----------
  align(rel_segments, audio, language, device) -> [{word,start,end}] or None
    rel_segments : objects with .start/.end/.text/.words (CLIP-RELATIVE seconds)
    audio        : float32 16kHz mono numpy array, clip-relative (t=0 = clip start)
    returns      : [{word,start,end}] clip-relative seconds, or None on failure
"""

import os
from dataclasses import dataclass

import numpy as np

os.environ["TORCHDYNAMO_DISABLE"]  = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"


_HF_MODELS = {
    "en": "facebook/wav2vec2-base-960h",
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

_DEFAULT_MODEL = "facebook/wav2vec2-large-960h-lv60-self"


def _load_model(language: str, device: str):
    """
    Returns (model, align_dict) where align_dict maps {char_lower: token_id}.
    The model outputs raw logits — caller applies log_softmax.
    """
    lang = (language or "en").lower().split("-")[0]
    hf_name = _HF_MODELS.get(lang, _DEFAULT_MODEL)

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(hf_name)
    model     = Wav2Vec2ForCTC.from_pretrained(hf_name).to(device)
    align_dict = {
        char.lower(): code
        for char, code in processor.tokenizer.get_vocab().items()
    }
    return model, align_dict


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


@dataclass
class _Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class _Span:
    token: int
    start: int
    end: int
    score: float


def _get_trellis(emission, tokens, blank_id):
    import torch
    num_frame  = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def _backtrack(trellis, emission, tokens, blank_id):
    import torch
    j = trellis.size(1) - 1
    t_start = int(torch.argmax(trellis[:, j]).item())

    path = []
    for t in range(t_start, 0, -1):
        stayed  = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        path.append(_Point(j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


def _merge_repeats(path, tokens):
    i1, i2 = 0, 0
    spans = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        spans.append(_Span(tokens[path[i1].token_index],
                           path[i1].time_index,
                           path[i2 - 1].time_index + 1,
                           score))
        i1 = i2
    return spans


def align(rel_segments, audio: np.ndarray, language: str, device: str):
    import torch

    words = _flatten_words(rel_segments)
    if not words:
        return None

    try:
        model, align_dict = _load_model(language, device)
    except Exception as e:
        print(f"[aligner] model load failed: {e}", flush=True)
        return None

    # blank token: '<pad>' / '[pad]' if present, else 0
    blank = align_dict.get("[pad]", align_dict.get("<pad>", 0))
    # space token: '|' for wav2vec2 models, ' ' for some others
    space = align_dict.get("|", align_dict.get(" ", None))

    try:
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        wav = wav.to(device)
        with torch.inference_mode():
            logits = model(wav).logits
        emission = torch.log_softmax(logits, dim=-1)[0].cpu().detach()  # (T, V)
    except Exception as e:
        print(f"[aligner] inference failed: {e}", flush=True)
        return None

    transcript = " ".join(w["word"] for w in words).lower()
    tokens = []
    for ch in transcript:
        if ch == " ":
            if space is not None:
                tokens.append(space)
        elif ch in align_dict:
            tokens.append(align_dict[ch])

    if not tokens:
        print("[aligner] empty token sequence — no recognised chars", flush=True)
        return None

    try:
        trellis = _get_trellis(emission, tokens, blank)
        path    = _backtrack(trellis, emission, tokens, blank)
        if path is None:
            print("[aligner] backtrack failed", flush=True)
            return None
        spans = _merge_repeats(path, tokens)
    except Exception as e:
        print(f"[aligner] forced align failed: {e}", flush=True)
        return None

    # wav2vec2 has a fixed stride of 320 samples at 16kHz = 50 fps
    fps = 16000 / 320

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

    # Map groups back to words. If group count doesn't match word count
    # (unknown chars dropped tokens), fall back to the input timestamps.
    result = []
    for i, wd in enumerate(words):
        if i < len(groups) and groups[i]:
            t_start = groups[i][0].start / fps
            t_end   = groups[i][-1].end  / fps
        else:
            t_start, t_end = wd["start"], wd["end"]
        result.append({"word": wd["word"], "start": t_start, "end": t_end})

    return result or None
