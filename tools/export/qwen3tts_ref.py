"""Reference activations for Qwen3-TTS parity tests (models/qwen3tts).

Runs the qwen-tts 0.1.1 reference (Qwen3-TTS-12Hz-0.6B-CustomVoice) on CPU
in float32 and saves, per case:

  tokenizer  text -> ids for a few strings through the model's processor
             (Qwen2TokenizerFast over vocab.json/merges.txt), and the backend
             tokenizer.json the processor actually runs.
  prompt     the talker's prefill input embeddings for one utterance.
  greedy     a greedy run (talker and code predictor both argmax; the
             talker's repetition penalty and suppressed tokens still apply):
             the talker's prefill logits, the first frame's code-predictor
             logits, and every frame's 16 codes.
  decode     the speech-tokenizer decoder on the greedy codes: the RVQ
             output, the pre-transformer output, and the waveform.

and, on the 1.7B checkpoints (VoiceDesign then Base — design a voice, then
clone it, the workflow that keeps one voice across many lines):

  design     VoiceDesign greedy from a text description: prompt, prefill and
             first-frame logits, codes, waveform.
  clone      Base, in-context cloning of the design waveform (reference
             transcript = the design text): the speaker encoder's mel and
             x-vector, the codec encoder's codes, the talker prompt and
             trailing text rows, greedy codes, waveform.

Needs its own venv (qwen-tts pins transformers==4.57.3):

    uv venv -p 3.12 tools/export/.venv-tts
    uv pip install --python tools/export/.venv-tts/bin/python torch torchaudio \
        transformers==4.57.3 accelerate==1.12.0 einops librosa soundfile sox onnxruntime
    uv pip install --python tools/export/.venv-tts/bin/python --no-deps qwen-tts==0.1.1
    HF_HUB_OFFLINE=1 tools/export/.venv-tts/bin/python tools/export/qwen3tts_ref.py [--design-clone | --resample | --all]

Writes testdata/qwen3tts/<name>.{in,out}.<i>.bin + <name>.json (same raw
little-endian layout as zoo.py). Outputs derive from the checkpoint and are
not committed.
"""

import json
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download

from qwen_tts import Qwen3TTSModel

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "qwen3tts")
REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DESIGN_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
INSTRUCT = "A calm, warm woman in her thirties with a clear, steady voice, speaking at a relaxed pace."
DESIGN_TEXT = "The lighthouse keeper counted every wave that night."
CLONE_TEXT = "Every line keeps the same voice."
# Longer than the reference's codes: the surplus text streams in as
# trailing rows during generation.
CLONE_LONG_TEXT = ("Every line keeps the same voice, from the first quiet greeting at dawn to the "
                   "last tired goodnight, because each one is cloned from a single kept reference.")

TEXT = "Hello from ingot, a pure Go inference runtime."
SPEAKER = "ryan"
LANGUAGE = "english"
FRAMES = 48  # max_new_tokens for the greedy run (~3.8 s of audio)


def save(name, ins, outs, meta):
    os.makedirs(OUT, exist_ok=True)
    man = {"meta": meta, "inputs": [], "outputs": []}
    for kind, arrs in (("in", ins), ("out", outs)):
        for i, (label, x) in enumerate(arrs):
            x = np.ascontiguousarray(x)
            f = f"{name}.{kind}.{i}.bin"
            x.tofile(os.path.join(OUT, f))
            man["inputs" if kind == "in" else "outputs"].append(
                {"name": label, "file": f, "dtype": str(x.dtype), "shape": list(x.shape)})
    json.dump(man, open(os.path.join(OUT, name + ".json"), "w"), indent=1)
    print(f"{name}: " + ", ".join(f"{l}{list(x.shape)}" for l, x in ins + outs))


def capture(model):
    """Hooks recording the talker's prompt, trailing text, logits."""
    talker = model.talker
    cap = {"prompt": None, "trailing": None, "talker_logits": [], "cp_logits": []}
    orig_generate = talker.generate

    def talker_generate(*a, **kw):
        cap["prompt"] = kw["inputs_embeds"].detach().clone()
        cap["trailing"] = kw["trailing_text_hidden"].detach().clone()
        return orig_generate(*a, **kw)

    talker.generate = talker_generate
    talker.codec_head.register_forward_hook(
        lambda _m, _i, out: cap["talker_logits"].append(out[0, -1].detach().clone()))
    cp = talker.code_predictor
    orig_cp_forward = cp.forward

    def cp_forward(*a, **kw):
        r = orig_cp_forward(*a, **kw)
        if len(cap["cp_logits"]) < 15:
            cap["cp_logits"].append(r.logits[0, -1].detach().clone())
        return r

    cp.forward = cp_forward
    return cap


GREEDY = dict(do_sample=False, subtalker_dosample=False, repetition_penalty=1.05,
              top_k=50, top_p=1.0, temperature=0.9,
              subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)


def design_and_clone():
    torch.manual_seed(0)
    tts = Qwen3TTSModel.from_pretrained(snapshot_download(DESIGN_REPO, local_files_only=True),
                                        dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    cap = capture(tts.model)
    with torch.no_grad():
        wavs, sr = tts.generate_voice_design(DESIGN_TEXT, INSTRUCT, language="english",
                                             max_new_tokens=FRAMES * 2, **GREEDY)
    ids = tts._tokenize_texts([tts._build_assistant_text(DESIGN_TEXT)])[0]
    meta = {"text": DESIGN_TEXT, "instruct": INSTRUCT, "language": "english", "frames": FRAMES * 2}
    codes = LAST["codes"]
    save("design", [("ids", ids[0].numpy().astype(np.int64))],
         [("embeds", cap["prompt"][0].float().numpy()),
          ("prefill_logits", cap["talker_logits"][0].float().numpy()),
          ("cp_logits", torch.stack(cap["cp_logits"]).float().numpy()),
          ("codes", codes),
          ("wav", np.asarray(wavs[0], dtype=np.float32))], meta)
    ref_wav = np.asarray(wavs[0], dtype=np.float32)
    del tts, cap

    tts = Qwen3TTSModel.from_pretrained(snapshot_download(BASE_REPO, local_files_only=True),
                                        dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    m = tts.model
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    with torch.no_grad():
        mel = mel_spectrogram(torch.from_numpy(ref_wav).unsqueeze(0), n_fft=1024, num_mels=128,
                              sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000)
        items = tts.create_voice_clone_prompt(ref_audio=(ref_wav, sr), ref_text=DESIGN_TEXT)
    cap = capture(m)
    with torch.no_grad():
        wavs, _ = tts.generate_voice_clone(CLONE_TEXT, language="english", voice_clone_prompt=items,
                                           max_new_tokens=FRAMES * 2, **GREEDY)
    ids = tts._tokenize_texts([tts._build_assistant_text(CLONE_TEXT)])[0]
    codes = LAST["codes"]
    save("clone", [("ref_wav", ref_wav), ("ids", ids[0].numpy().astype(np.int64))],
         [("mel", mel[0].float().numpy()),
          ("xvector", items[0].ref_spk_embedding.float().numpy()),
          ("ref_codes", items[0].ref_code.numpy().astype(np.int64)),
          ("embeds", cap["prompt"][0].float().numpy()),
          ("trailing", cap["trailing"][0].float().numpy()),
          ("prefill_logits", cap["talker_logits"][0].float().numpy()),
          ("codes", codes),
          ("wav", np.asarray(wavs[0], dtype=np.float32))],
         {"text": CLONE_TEXT, "ref_text": DESIGN_TEXT, "language": "english", "frames": FRAMES * 2})

    # Long target text (trailing text rows), then x-vector-only cloning.
    for name, text, prompt in (("clone_long", CLONE_LONG_TEXT, items), ("clone_xvec", CLONE_TEXT, None)):
        if prompt is None:
            with torch.no_grad():
                prompt = tts.create_voice_clone_prompt(ref_audio=(ref_wav, sr), x_vector_only_mode=True)
        cap = capture(m)
        with torch.no_grad():
            tts.generate_voice_clone(text, language="english", voice_clone_prompt=prompt,
                                     max_new_tokens=24, **GREEDY)
        ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
        save(name, [("ids", ids[0].numpy().astype(np.int64))],
             [("embeds", cap["prompt"][0].float().numpy()),
              ("trailing", cap["trailing"][0].float().numpy()),
              ("codes", LAST["codes"])],
             {"text": text, "ref_text": DESIGN_TEXT if name == "clone_long" else "", "language": "english", "frames": 24})


# LAST["codes"]: the codes [F, 16] of the most recent model.generate()
# (patched in __main__; the wrappers return only waveforms).
LAST = {}


def resample_cases():
    """librosa.resample (soxr_hq, what qwen-tts runs on reference clips that
    are not 24 kHz) on a speech-band test signal at common input rates."""
    import librosa
    rng = np.random.default_rng(0)
    for sr in (16000, 22050, 44100, 48000):
        n = sr * 3 // 2
        t = np.arange(n) / sr
        # Chirp 80 Hz -> 7 kHz plus band-limited noise and a few tones.
        x = 0.3 * np.sin(2 * np.pi * (80 * t + (7000 - 80) / (2 * t[-1]) * t * t))
        for f in (220, 1000, 3300):
            x += 0.1 * np.sin(2 * np.pi * f * t)
        x += 0.05 * rng.standard_normal(n)
        x = x.astype(np.float32)
        y = librosa.resample(x, orig_sr=sr, target_sr=24000).astype(np.float32)
        save(f"resample_{sr}", [("x", x)], [("y", y)], {"orig_sr": sr, "target_sr": 24000})


def main():
    torch.manual_seed(0)
    snap = snapshot_download(REPO, local_files_only=True)
    tts = Qwen3TTSModel.from_pretrained(snap, dtype=torch.float32, device_map="cpu",
                                        attn_implementation="eager")
    model = tts.model.eval()
    talker = model.talker

    # --- tokenizer --------------------------------------------------------
    tok = tts.processor.tokenizer
    os.makedirs(OUT, exist_ok=True)
    tok.backend_tokenizer.save(os.path.join(OUT, "backend_tokenizer.json"))
    cases = [
        TEXT,
        tts._build_assistant_text(TEXT),
        "Numbers 1234567, punctuation!? — and   spaces\n\nnewlines.",
        "你好，世界。こんにちは 안녕하세요 Ünïcödé café",
        "<|im_start|>user\nSpeak slowly.<|im_end|>\n",
    ]
    json.dump({"cases": [{"text": t, "ids": tok(t)["input_ids"]} for t in cases]},
              open(os.path.join(OUT, "tokenizer.json"), "w"), indent=1, ensure_ascii=False)
    print(f"tokenizer: {len(cases)} cases")

    # --- capture hooks ----------------------------------------------------
    cap = {"prompt": None, "talker_logits": [], "cp_logits": [], "cp_calls": 0}
    orig_generate = talker.generate

    def talker_generate(*a, **kw):
        cap["prompt"] = kw["inputs_embeds"].detach().clone()
        return orig_generate(*a, **kw)

    talker.generate = talker_generate

    def head_hook(_m, _i, out):
        cap["talker_logits"].append(out[0, -1].detach().clone())

    talker.codec_head.register_forward_hook(head_hook)
    cp = talker.code_predictor
    orig_cp_forward = cp.forward

    def cp_forward(*a, **kw):
        r = orig_cp_forward(*a, **kw)
        if len(cap["cp_logits"]) < 15:
            cap["cp_logits"].append(r.logits[0, -1].detach().clone())
        return r

    cp.forward = cp_forward

    ids = tts._tokenize_texts([tts._build_assistant_text(TEXT)])[0]
    with torch.no_grad():
        codes_list, _ = model.generate(
            input_ids=[ids], instruct_ids=[None], languages=[LANGUAGE], speakers=[SPEAKER],
            non_streaming_mode=True, max_new_tokens=FRAMES,
            do_sample=False, subtalker_dosample=False, repetition_penalty=1.05,
            top_k=50, top_p=1.0, temperature=0.9,
            subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
    codes = codes_list[0]  # [F, 16]

    meta = {"text": TEXT, "speaker": SPEAKER, "language": LANGUAGE, "frames": FRAMES}
    save("prompt", [("ids", ids[0].numpy().astype(np.int64))],
         [("embeds", cap["prompt"][0].float().numpy())], meta)
    save("greedy", [("ids", ids[0].numpy().astype(np.int64))],
         [("prefill_logits", cap["talker_logits"][0].float().numpy()),
          ("step1_logits", cap["talker_logits"][1].float().numpy()),
          ("cp_logits", torch.stack(cap["cp_logits"]).float().numpy()),
          ("codes", codes.numpy().astype(np.int64))], meta)

    # --- speech-tokenizer decoder ----------------------------------------
    dec = model.speech_tokenizer.model.decoder
    inter = {}
    dec.pre_conv.register_forward_hook(lambda m, i, o: inter.__setitem__("rvq", i[0].detach().clone()))
    dec.pre_transformer.register_forward_hook(
        lambda m, i, o: inter.__setitem__("pre_tf", o.last_hidden_state.detach().clone()))
    with torch.no_grad():
        wav = dec(codes.T.unsqueeze(0))[0, 0]
        wavs, sr = model.speech_tokenizer.decode([{"audio_codes": codes}])
    save("decode", [("codes", codes.numpy().astype(np.int64))],
         [("rvq", inter["rvq"][0].float().numpy()),
          ("pre_tf", inter["pre_tf"][0].float().numpy()),
          ("wav", wav.float().numpy()),
          ("wav_trimmed", np.asarray(wavs[0], dtype=np.float32))], {"sample_rate": sr})


if __name__ == "__main__":
    import sys
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    _orig = Qwen3TTSForConditionalGeneration.generate

    def _gen(self, *a, **kw):  # remember the codes each generate() returns
        codes, hid = _orig(self, *a, **kw)
        LAST["codes"] = codes[0].numpy().astype(np.int64)
        return codes, hid

    Qwen3TTSForConditionalGeneration.generate = _gen
    if "--resample" in sys.argv or "--all" in sys.argv:
        resample_cases()
        if "--resample" in sys.argv:
            sys.exit(0)
    if "--design-clone" in sys.argv or "--all" in sys.argv:
        design_and_clone()
    if "--design-clone" not in sys.argv:
        main()
