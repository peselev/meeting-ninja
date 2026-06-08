from __future__ import annotations
import json
from pathlib import Path


def is_pyannote_available() -> bool:
    try:
        import pyannote.audio  # noqa: F401
        return True
    except ImportError:
        return False


def diarize(audio_path: str, hf_token: str) -> list[dict]:
    """
    Run pyannote speaker diarization.
    Returns list of {start_sec, end_sec, label} sorted by start_sec.
    Raises if pyannote not installed, token invalid, or licenses not accepted.
    """
    import logging
    log = logging.getLogger("diarize")
    from pyannote.audio import Pipeline

    # ── PyTorch 2.6+ compatibility ───────────────────────────────────────────
    # torch 2.6 flipped torch.load's default to weights_only=True, which refuses
    # to unpickle pyannote's checkpoints (they embed a TorchVersion object).
    # We trust these checkpoints (gated HF models), so we both allowlist the
    # offending globals AND force weights_only=False as a belt-and-suspenders
    # fallback for older/newer torch builds.
    try:
        import torch
        # Allowlist the specific globals pyannote/torch embed in checkpoints.
        try:
            from torch.serialization import add_safe_globals
            safe = []
            try:
                from torch.torch_version import TorchVersion
                safe.append(TorchVersion)
            except Exception:
                pass
            try:
                from pyannote.audio.core.task import Specifications
                safe.append(Specifications)
            except Exception:
                pass
            if safe:
                add_safe_globals(safe)
                log.debug("Added %d classes to torch safe globals.", len(safe))
        except Exception as e:
            log.debug("add_safe_globals unavailable or failed: %s", e)

        # FORCE weights_only=False. lightning_fabric passes weights_only=True
        # explicitly, so setdefault isn't enough — we overwrite it. Safe here:
        # these are gated HF checkpoints we authenticated to and trust.
        if not getattr(torch.load, "_mn_patched", False):
            _orig_torch_load = torch.load
            def _patched_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)
            _patched_load._mn_patched = True
            torch.load = _patched_load
            log.debug("Patched torch.load to force weights_only=False.")
    except Exception as e:
        log.debug("torch compatibility shim skipped: %s", e)

    # Authenticate once via huggingface_hub.login (writes token to cache).
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            log.debug("Authenticated with HuggingFace via login().")
        except Exception as e:
            log.warning("huggingface_hub.login failed (%s); trying anyway.", e)

    # Try the current community-1 pipeline (pyannote 4.x) first, then fall back
    # to the legacy 3.1 pipeline. Pass token= when available (4.x accepts it).
    pipeline = None
    last_err = None
    for model_name in ("pyannote/speaker-diarization-community-1",
                       "pyannote/speaker-diarization-3.1"):
        for kwargs in ({"token": hf_token} if hf_token else {}, {}):
            try:
                pipeline = Pipeline.from_pretrained(model_name, **kwargs)
                if pipeline is not None:
                    log.debug("Loaded pipeline: %s", model_name)
                    break
            except TypeError as e:
                last_err = e  # kwarg mismatch — try next form
                continue
            except Exception as e:
                last_err = e
                break  # different error — try next model
        if pipeline is not None:
            break

    if pipeline is None:
        raise RuntimeError(
            "Could not load any pyannote pipeline. Most likely the model "
            "license wasn't accepted. Visit and click 'Agree' on the model "
            "page for whichever you're using:\n"
            "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            f"Last error: {last_err}"
        )

    # Feed pyannote an in-memory waveform loaded via soundfile, bypassing
    # torchaudio's FFmpeg backend (which is broken on some macOS installs:
    # "libavutil…dylib not loaded / no LC_RPATH's found"). Our audio is always
    # a plain 16kHz mono WAV from ffmpeg, which soundfile reads cleanly.
    #
    # ProgressHook prints diarization stages with a progress bar so the run
    # isn't a silent black box (diarization can take several minutes on CPU).
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        _hook_cls = ProgressHook
    except Exception:
        _hook_cls = None

    def _run(input_arg):
        if _hook_cls is not None:
            with _hook_cls() as hook:
                return pipeline(input_arg, hook=hook)
        return pipeline(input_arg)

    try:
        import soundfile as sf
        import torch
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        # soundfile gives (frames, channels); pyannote wants (channels, frames)
        waveform = torch.from_numpy(waveform.T)
        log.debug("Loaded audio via soundfile: %s frames @ %d Hz",
                  waveform.shape[-1], sample_rate)
        log.info("      Running diarization (this can take a few minutes)…")
        diarization = _run({"waveform": waveform, "sample_rate": sample_rate})
    except Exception as e:
        log.debug("soundfile path failed (%s); falling back to file path.", e)
        diarization = _run(audio_path)

    # Extract speaker turns, supporting both APIs:
    #  - pyannote 4.x: result has .speaker_diarization, iterated as (turn, speaker)
    #  - pyannote 3.x: result is an Annotation, use .itertracks(yield_label=True)
    turns = []
    annotation = getattr(diarization, "speaker_diarization", None)
    if annotation is not None:
        # 4.x: iterate (turn, speaker); also supports itertracks for safety
        try:
            for turn, speaker in annotation:
                turns.append({"start_sec": turn.start, "end_sec": turn.end,
                              "label": str(speaker)})
        except (TypeError, ValueError):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                turns.append({"start_sec": turn.start, "end_sec": turn.end,
                              "label": str(speaker)})
    else:
        # 3.x: the result itself is the Annotation
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append({"start_sec": turn.start, "end_sec": turn.end,
                          "label": str(speaker)})

    return sorted(turns, key=lambda t: t["start_sec"])


def merge_diarization_with_segments(
    whisper_segments: list[dict],
    diarization_turns: list[dict],
) -> list[dict]:
    """
    Assign a speaker label to each Whisper segment by finding the
    diarization turn with the greatest time overlap.

    whisper_segments: [{start_sec, end_sec, text}, ...]
    diarization_turns: [{start_sec, end_sec, label}, ...]
    Returns: [{start_sec, end_sec, text, speaker_label}, ...]
    """
    result = []
    for seg in whisper_segments:
        best_label = None
        best_overlap = 0.0
        for turn in diarization_turns:
            overlap = min(seg["end_sec"], turn["end_sec"]) - max(seg["start_sec"], turn["start_sec"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn["label"]
        result.append({**seg, "speaker_label": best_label})
    return result


def collect_speaker_samples(
    merged_segments: list[dict],
    n_samples: int = 3,
) -> dict[str, list[dict]]:
    """
    For each unique speaker label, collect up to n_samples segments
    as representative examples for the labeling UI.
    Returns {label: [{start_sec, end_sec, text}, ...]}
    """
    from collections import defaultdict
    samples: dict[str, list] = defaultdict(list)
    for seg in merged_segments:
        label = seg.get("speaker_label") or "UNKNOWN"
        if len(samples[label]) < n_samples:
            samples[label].append({
                "start_sec": seg["start_sec"],
                "end_sec":   seg["end_sec"],
                "text":      seg["text"],
            })
    return dict(samples)
