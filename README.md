<div align="center">
<h1>Fish Speech</h1>

**English** | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | [العربية](docs/README.ar.md) | [Español](docs/README.es.md)  <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
<br>
</div>
<br>

<div align="center">
    <img src="https://count.getloli.com/get/@fish-speech?theme=asoul" /><br>
</div>

<br>

<div align="center">
    <a target="_blank" href="https://discord.gg/Es5qTB9BcN">
        <img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
    </a>
    <a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
        <img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
    </a>
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

<div align="center">
    <a target="_blank" href="https://huggingface.co/fishaudio/drama-3">
        <img alt="HuggingFace Drama 3" src="https://img.shields.io/badge/🤗%20-Drama%203-orange"/>
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace S2-Pro" src="https://img.shields.io/badge/🤗%20-S2--Pro-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://arxiv.org/abs/2610.18010">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **License Notice**  
> This codebase and its associated model weights are released under **[FISH AUDIO COMMUNITY LICENSE](LICENSE)**. The S2-Pro weights remain under the Fish Audio Research License. Please refer to [LICENSE](LICENSE) for more details. We will take action against any violation of the license.

> [!WARNING]
> **Legal Disclaimer**  
> We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Quick Start

### For Human

- [Installation](https://speech.fish.audio/install/): environment setup, download weights, Docker deployment
- [Inference](https://speech.fish.audio/inference/): command line, API server, WebUI

> [!IMPORTANT]
> **We strongly recommend SGLang-Omni for users who care about efficiency; see the [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_drama_3/README.md).**
>
> **For vLLM Omni server, please read [vLLM-Omni Fish Speech S2 Pro Recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/fishaudio/Fish-Speech-S2-Pro.md) and the [User Guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/examples/online_serving/text_to_speech.md#fish-speech-s2-pro).**

### FastAPI Server and Client

This fork keeps its local S2-Pro service on top of upstream `5415669` (Drama-3).
The original eight local commits are preserved at
`backup/main-before-upstream-2026-10-26` (`3b74d34`). Local `main` tracks
`upstream/main`; development continues on `feature/local-tts-upstream-2026-10-26`.
The active KV cache optimization is already included upstream. The removed
legacy Gradio UI remains available on the backup branch.

The local service retains S2-Pro repetition penalties, iterative prompts,
CPU reference caching, MPS cleanup, graceful shutdown, and performance diagnostics.
Drama-3 is available through upstream's `tools/api_server.py` and requires CUDA;
the command below continues to use your existing S2-Pro weights on MPS.
The upstream API uses its new sampling defaults when fields are omitted; the
local FastAPI client retains its explicit generation defaults.

Start the local FastAPI server:

```bash
fish-tts-server \
  --host 127.0.0.1 \
  --port 8002 \
  --llama-checkpoint-path ./modelsweights/s2-pro \
  --decoder-checkpoint-path ./modelsweights/s2-pro/codec.pth \
  --storage-root ./storage/fish_speech_service \
  --device mps \
  --dtype float16 \
  --max-seq-len 4096
```


```bash
fish-tts-server \
  --host 127.0.0.1 \
  --port 8002 \
  --llama-checkpoint-path ./modelsweights/s2-pro \
  --decoder-checkpoint-path ./modelsweights/s2-pro/codec.pth \
  --storage-root ./storage/fish_speech_service \
  --device mps \
  --dtype float16 \
  --max-seq-len 4096 \
  --compile \
  --compile-backend inductor
```

The FastAPI server and client default to port `8002`. On macOS, the server reads `.env` before importing PyTorch and defaults to conservative MPS memory settings:
`PYTORCH_ENABLE_MPS_FALLBACK=1`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.55`, and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.75`.

For S2-Pro on Apple MPS, append **both** `--compile --compile-backend inductor`
to the command above to compile autoregressive decode. This keeps the growing
active KV prefix (`dynamic=True`) and uses native MPS attention and repetition
window detection through small compiler wrappers; neither operation moves to CPU.
The codec remains eager. Compilation is optional and disabled by default.

On an M4 Pro with 48 GiB unified memory, PyTorch 2.8.0, FP16 and a 4096-token
cache, a Japanese voice-cloning benchmark produced these results (three warm
requests averaged, identical text/reference/seed/sampling parameters, capped at
96 semantic frames / 4.46 seconds of audio):

| Decode mode | Semantic frames/s | Request wall time |
| --- | ---: | ---: |
| Eager | 7.20 | 13.97 s |
| `--compile --compile-backend aot_eager` | 4.90 | 20.35 s |
| `--compile --compile-backend inductor` | 11.27 | 9.52 s |

Inductor improved semantic throughput by about 57% and reduced request time by
about 32% in this workload. Post-request MPS driver allocation was about 14.33 GiB
for both eager and Inductor; this is not peak memory. The first Inductor synthesis
took 51 seconds, excluding 56 seconds to load the models. Allow a longer client
timeout for the first request. New shapes/settings can trigger additional
compilation, so compare warmed requests. A fixed seed does not guarantee identical
audio between eager and compiled execution.

`--compile` alone preserves the previous backend selection: `inductor` on CUDA,
`aot_eager` on MPS/CPU. Select `inductor` explicitly for the measured MPS speedup;
`aot_eager` does not provide the same kernel fusion. Equivalent `.env` settings
are `FISH_TTS_COMPILE=true` and `FISH_TTS_COMPILE_BACKEND=inductor`.
Use `--no-compile` to override the environment and return to eager execution.
`GET /fishspeech/health` reports the configured `compile` and `compile_backend`;
actual compilation is lazy and happens during the first decode.
Do not combine compilation with `--perf-detail`, `FISH_TTS_PERF_DETAIL=true`, or
`FISH_TTS_DEBUG=true`. Drama-3 compilation is not supported by this service.

API-key authentication is disabled when `FISH_TTS_API_KEY` is unset or empty.
To enable it on a remote server, set this in that server's `.env` and restart;
the server launch command stays the same:

```dotenv
FISH_TTS_API_KEY=replace-with-a-long-random-secret
```

Both `/fishspeech/*` and `/v1/*` then require either
`Authorization: Bearer <key>` or `api-key: <key>`. This includes health checks,
reference/output downloads, synthesis and shutdown, even on loopback connections
arriving through an SSH tunnel. Shutdown still requires a loopback connection and
its confirmation header. `/docs` and `/openapi.json` remain accessible.

`fish-tts-client` and `FishSpeechHttpxClient` read `FISH_TTS_API_KEY` from their
environment or project `.env`, including for automatic audio downloads. Override
it with `fish-tts-client --api-key <key> health` or
`FishSpeechHttpxClient(api_key="...")`. Prefer environment configuration to avoid
putting a secret in shell history. The Python client sends its key only to the
configured service origin. For a manual authenticated health check:

```bash
curl http://127.0.0.1:8002/fishspeech/health \
  -H "Authorization: Bearer $FISH_TTS_API_KEY"
```

The shell variable in this curl example must be exported separately; curl does
not read `.env`. Use an encrypted SSH/frp tunnel or HTTPS for remote connections.

Synthesize with a reference audio and transcript:

```bash
fish-tts-client voice-clone \
  --ref-audio inputs/ref.wav \
  --ref-text "Reference transcript text" \
  --text "Text to synthesize" \
  --repetition-penalty 1.1 \
  --download-to outputs/out.wav
```

Long text is split into iterative prompt batches by default, using
`--chunk-length` as the UTF-8 byte limit. Pass `--no-iterative-prompt` to
synthesize the request as one batch. The equivalent FastAPI form field is
`iterative_prompt=true|false`; `/v1/tts` accepts the same field in
`ServeTTSRequest`.

For repeated requests with the same reference audio, add
`--use-memory-cache on`. The cache is keyed by the reference audio bytes,
stores only the encoded VQ tokens on CPU, and lasts until the server exits.
Single voice-clone requests default to `off`; batch-file requests default to
`on`.

Inference logs report phase wall timings, CUDA accelerator-event timings,
semantic frame latency/rate, end-to-end RTF, and MPS or CUDA allocator memory.
MPS metrics deliberately avoid event elapsed-time reads because they can block
indefinitely in the threaded worker on PyTorch 2.8; default MPS metrics add no
frame-level synchronization. Actual Metal timing, bandwidth, and GPU occupancy
require a Metal trace; start the server with `--mps-profile` to emit PyTorch MPS
signposts for Instruments:

```bash
fish-tts-server ... --device mps --mps-profile
```

To diagnose semantic generation itself, run a separate benchmark server with
sampled stage timing:

```bash
fish-tts-server ... --device mps --perf-detail --perf-sample-frames 16
```

The equivalent project-scoped debug environment switch is:

```bash
FISH_TTS_DEBUG=true fish-tts-server ... --device mps
```

Use `FISH_TTS_PERF_SAMPLE_FRAMES=16` to change its sample limit. A bare
`DEBUG=true` is intentionally ignored so unrelated frameworks or IDE launch
settings cannot accidentally enable synchronized inference diagnostics.

`--perf-detail` synchronizes only selected semantic frames and reports
`slow_transformer`, `output_projection`, `slow_ar+output`, `main_sampler`, and
`fast_ar+residual_sampler`. It is diagnostic-only and changes the sampled
request's timing, so do not use that request's end-to-end RTF as a production
benchmark. Run `--mps-profile` separately, without `--perf-detail`, when
collecting an Instruments trace of GPU activity, occupancy, and bandwidth.

Before starting another memory-intensive local job, gracefully stop the server
and wait until port `8002` is no longer responding:

```bash
fish-tts-client shutdown
```

The shutdown API only accepts loopback clients and requires a confirmation
header supplied automatically by `fish-tts-client`. Because shutdown exits the
server process, start `fish-tts-server` again before the next synthesis session.
The server does not shut down automatically when it is idle.

If the virtual environment is not activated, use `.venv/bin/fish-tts-server` and `.venv/bin/fish-tts-client`.

### OpenAI-compatible and MiMo-style TTS APIs

The same `fish-tts-server` also serves `POST /v1/audio/speech` (binary audio),
`POST /v1/chat/completions` (MiMo-style audio JSON or PCM SSE), and
`GET /v1/models`. All routes share the existing model, inference lock, and
graceful shutdown controller.

```bash
curl http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"s2-pro","input":"你好，欢迎使用语音服务。","voice":"default","response_format":"wav"}' \
  --output speech.wav
```

`default` generates without a voice reference. For a reproducible reference
voice, put one WAV/MP3 and its matching `.lab` transcript in
`storage/fish_speech_service/voices/<voice-id>/`, then use that ID as `voice`.
Set `FISH_TTS_API_KEY` to optionally require Bearer or `api-key` authentication
on both `/v1/*` and `/fishspeech/*`.

See [中文接口说明与 OpenAI SDK / SSE 示例](docs/zh/openai_api.md) for voice cloning,
supported fields, audio formats, and the differences from OpenAI and MiMo.

### For LLM Agent

```
Install and configure Fish Audio Drama 3 (or S2-Pro) by following the instructions here: https://speech.fish.audio/install/
```

## Models

| Model | Size | Weights | Description |
|------|------|-------------|-------------|
| **Drama 3** | 35B-A3B (3B active per token) | [HuggingFace](https://huggingface.co/fishaudio/drama-3) | New-generation flagship for multi-character scenes and long content, with multi-turn revision and local editing |
| S2-Pro | 4B parameters | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | Previous-generation flagship with faster inference |

The model type is detected from the checkpoint's `config.json`, and both share the same command line and API server.

---

## Fish Audio Drama 3

**The new-generation voice model, upgraded across the board over S2-Pro.**

Like the S2 series, Drama 3 supports multi-speaker generation, voice cloning and natural-language control, but it makes clear advances in long-form audio, interaction and editability.

### Key features

- **Multi-turn director revision**: guide the previous take directly ("gentler, and a bit sadder"), and the model regenerates it with that context instead of starting over.
- **Infilling and precise editing**: edit audio the way you edit text — replace a single word ("Wednesday" → "Thursday") while the surrounding timbre and transitions stay natural.
- **Scene-style multi-character production**: voice a complete scene with several characters straight from a script description, with streaming support for long text.
- **Long dialogues and long speech**: built for long recordings and extended sessions, keeping characters and timbre stable throughout.
- **Stronger expressiveness**: laughter, crying, whispers and dramatic emotion are no longer flattened by the reference, so the delivery is much closer to a real performance.
- **Multilingual voice cloning**: sets a new level of SOTA for cross-lingual cloning error rates (CER / WER).
- **Natural-language tag control**: inline tags such as `[whisper]`, `[angry]` and `[laughing]` control emotion and prosody.

### What changed compared with S2-Pro

| Capability | S2-Pro | Drama 3 |
| --- | --- | --- |
| Multi-speaker | Supported | Supported, as scene-style multi-character production |
| Voice cloning | Supported | Supported, with lower multilingual error rates |
| Inline natural-language instructions | Supported | Supported |
| Local editing / infilling | Not supported | Supported |
| Multi-turn director revision | Not supported | Supported |
| Long content | Chunked generation | Long dialogues and long speech, with streaming |
| Strong emotion (laughter, crying, whispers) | Moderate | Stronger |

### Results

Drama 3 takes a clear leap over S2 on public benchmarks: multilingual voice cloning (MiniMax Multilingual Testset), free-form speech editing (Ming-Freeform) and multi-turn emotion revision (Step-Audio-Edit), where control precision, multilingual error rates (CER / WER) and style accuracy all surpass the previous generation.

Read the [technical report](https://arxiv.org/abs/2610.18010) for the architecture, the training recipe and the full evaluation results.

---

## Fish Audio S2-Pro

Our previous-generation flagship: natural-language controllable and emotionally expressive speech, about a hundred languages, and small enough to deploy on a consumer GPU.

- **Inline natural-language instructions**: tags such as `[pause]`, `[excited]` and `[whisper]` control prosody and emotion anywhere in the text.
- **Native multi-speaker generation**: speaker IDs separate the characters inside one text, so you no longer generate and concatenate one speaker at a time.
- **Rapid voice cloning**: 10–30 seconds of reference audio is enough to reproduce a timbre.
- **Single pass**: long audio can be streamed out in one go.
- **Production streaming**: RTF 0.195 on a single H200, about 100 ms to first audio.

Visit the [Fish Audio website](https://fish.audio/) for a live playground, and read the [blog post](https://fish.audio/blog/fish-audio-open-sources-s2/) and [technical report](https://arxiv.org/abs/2603.08823) for more details.

<details>
<summary><b>S2-Pro benchmark results (click to expand)</b></summary>

| Benchmark | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER (Chinese) | **0.54%** (best overall) |
| Seed-TTS Eval — WER (English) | **0.99%** (best overall) |
| Audio Turing Test (with instruction) | **0.515** posterior mean |
| EmergentTTS-Eval — Win Rate | **81.88%** (highest overall) |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — Quality | **4.51 / 5.0** |
| Multilingual (MiniMax Testset) — Best WER | **11 of 24** languages |
| Multilingual (MiniMax Testset) — Best SIM | **17 of 24** languages |

On Seed-TTS Eval, S2 achieves the lowest WER among all evaluated models including closed-source systems: Qwen3-TTS (0.77/1.24), MiniMax Speech-02 (0.99/1.90), Seed-TTS (1.12/2.25). On the Audio Turing Test, 0.515 surpasses Seed-TTS (0.417) by 24% and MiniMax-Speech (0.387) by 33%. On EmergentTTS-Eval, S2 achieves particularly strong results in paralinguistics (91.61% win rate), questions (84.41%), and syntactic complexity (83.39%).

<img src="./docs/assets/totalability.png" width=200%>

</details>

## Credits

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Tech Report
```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}

@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823}, 
}

@misc{liao2026fishaudiodrama3technical,
      title={Fish Audio Drama 3 Technical Report}, 
      author={Dawei Han and Shidong Li and Shijia Liao and Jiahua Liu and Songting Liu and Qingzheng Wang and Yuxuan Wang and Xin Chen and Yifan Cheng and Tianyu Li and Gangin Park and Xuehao Yang and Chang Zeng and Haoyang Zhang and Ruoyi Zhang and Xiangyu Zhang},
      year={2026},
      eprint={2610.18010},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2610.18010}, 
}
```
