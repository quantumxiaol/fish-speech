# 本地 OpenAI 兼容 TTS 与 MiMo 风格音频接口

沿用 README 中的 `fish-tts-server` 启动命令。接口位于同一个端口，复用同一个
S2-Pro 模型、参考音频缓存、推理锁和优雅关闭控制器。现有 `/fishspeech/...`
接口继续可用。

| 地址 | 输入 | 输出 |
| --- | --- | --- |
| `GET /v1/models` | 无 | OpenAI 格式的本地模型列表 |
| `POST /v1/audio/speech` | `model`、`input`、`voice` | 音频文件内容 |
| `POST /v1/chat/completions` | `model`、`messages`、`audio` | MiMo 风格的 Base64 音频 JSON；`stream:true` 时输出 SSE |

标准 Speech 路径及字段依据 [OpenAI Create speech](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create)。
Chat 接口采用 MiMo TTS 的“assistant 消息指定朗读文本”约定。它执行语音合成，
不生成聊天回答；这与 OpenAI 通用 Chat Completions 的对话生成语义不同。

## 模型、鉴权与音色

默认模型 ID 是 `s2-pro`，可通过 `--model-id` 或 `FISH_TTS_MODEL_ID` 修改。
这仅修改客户端使用的名称；实际权重仍由 `--llama-checkpoint-path` 决定。
未知模型 ID 返回 HTTP 400，不会转发到 OpenAI 或 MiMo 云端。

默认不要求密钥。设置 `FISH_TTS_API_KEY` 后，所有 `/v1/*` 请求需要以下任意一个请求头：

```text
Authorization: Bearer <你的本地服务密钥>
api-key: <你的本地服务密钥>
```

该设置只作用于新增的 `/v1/*`，原有 `/fishspeech/...` 鉴权规则保持不变。
OpenAI SDK 要求配置 `api_key`；未启用服务鉴权时，可填 `local`。

`voice="default"` 表示不使用参考音频，不代表某个固定说话人。要使用参考音色，
准备以下目录；音色 ID 使用英文字母、数字、下划线和连字符：

```text
storage/fish_speech_service/voices/my_voice/
├── reference.wav
└── reference.lab
```

每个音色目录包含一个 WAV 或 MP3 文件及同名 `.lab` 文件。
`.lab` 为非空 UTF-8 文本，内容是参考音频的准确转录。参考音频最大 25 MiB。
请求时指定 `voice="my_voice"`。可以用 `--voices-dir` 或 `FISH_TTS_VOICES_DIR`
指定另一处音色根目录。音色文件在每次请求时读取，编码结果按音频内容复用 CPU 缓存。

服务没有内置 OpenAI 的 `alloy`、`coral` 或 MiMo 的预置音色。
这些名称只有在你创建了对应本地音色目录后才能使用。

## 标准 Speech API

```bash
curl http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "s2-pro",
    "input": "你好，这是一段测试语音。",
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0
  }' --output speech.wav
```

| 字段 | 支持范围 |
| --- | --- |
| `model` | 必填；默认 `s2-pro` |
| `input` | 必填；1–4096 字符，不能全为空白 |
| `voice` | 必填；`default`、本地音色 ID，或 `{"id":"my_voice"}` |
| `response_format` | 默认 `mp3`；支持 `mp3`、`wav`、`flac`、`opus`、`aac`、`pcm` |
| `speed` | 默认 1.0；0.25–4.0，生成后使用 FFmpeg 调速并保持音高 |
| `stream_format` | 目前支持默认的 `audio`；`sse` 返回明确的 HTTP 400 |
| `instructions` | 暂不支持非空自然语言指令；可在 `input` 中使用 S2-Pro 支持的表达标签 |
| `fish_speech` | 可选的本地扩展参数，见后文 |

默认 MP3、WAV、FLAC、Opus 使用 libsndfile 编码。AAC 与非 1.0 语速需要服务器
安装 `ffmpeg`；macOS 可以用 `brew install ffmpeg`。缺少编码能力时，服务会在
推理前返回明确错误。

PCM 是 **24 kHz、单声道、16 位有符号小端，无文件头**；Opus 转为 48 kHz。
WAV、FLAC、MP3 使用解码器的采样率。Speech 接口完成合成后返回整段音频；
SDK 的 `with_streaming_response` 可用于下载，但不会让底层变成逐 token 音频输出。

OpenAI Python SDK 示例（客户端环境安装 `openai`）：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8002/v1",
    api_key="local",  # 启用鉴权时替换为本地服务密钥
    timeout=300,
)

with client.audio.speech.with_streaming_response.create(
    model="s2-pro",
    input="你好，这是一段测试语音。",
    voice="default",
    response_format="wav",
) as response:
    response.stream_to_file("speech.wav")
```

## MiMo 风格的 Chat 音频 API

请求中提供 **恰好一条 assistant 消息**，其内容就是要合成的文字。
`audio.voice` 可省略，默认 `default`；非流式默认输出 WAV。

```python
import base64
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8002/v1", api_key="local", timeout=300)
response = client.chat.completions.create(
    model="s2-pro",
    messages=[{"role": "assistant", "content": "你好，这是一段测试语音。"}],
    audio={"voice": "default", "format": "wav"},
)
Path("chat-speech.wav").write_bytes(
    base64.b64decode(response.choices[0].message.audio.data)
)
```

响应中 `object` 为 `chat.completion`，音频在
`choices[0].message.audio.data` 中。`audio.transcript` 为实际朗读的原文，
`audio.expires_at` 和 `usage` 为 `null`，服务不虚构 token 用量。
`finish_reason="length"` 表示至少一个文本分段达到生成上限，音频可能被截断；
正常结束时为 `stop`。

支持的 `audio.format` 为 `wav`、`mp3`、`pcm`、`pcm16`。
`pcm16` 是 `pcm` 的别名。

### Base64 参考音频克隆

除本地音色 ID 外，Chat 的 `audio.voice` 支持 WAV/MP3 文件的原始 Base64 字符串。
Fish Speech 需要参考音频转录，因此必须同时提供扩展参数
`fish_speech.reference_text`。服务不自动转录参考音频。

```python
reference = base64.b64encode(Path("inputs/ref.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="s2-pro",
    messages=[{"role": "assistant", "content": "需要合成的新内容。"}],
    audio={"voice": reference, "format": "wav"},
    extra_body={"fish_speech": {"reference_text": "参考音频的准确转录。"}},
)
```

### SSE 流式输出

`stream:true` 时，省略 `audio.format` 或指定 `pcm` / `pcm16`。
每个 SSE `data:` 对象的类型为 `chat.completion.chunk`，音频增量位于
`choices[0].delta.audio.data`，内容是 Base64 PCM。每段须分别 Base64 解码，
再拼接二进制数据；最后发送结束原因和 `data: [DONE]`。

```python
with client.chat.completions.create(
    model="s2-pro",
    messages=[{"role": "assistant", "content": "第一段内容。第二段内容。"}],
    audio={"voice": "default", "format": "pcm16"},
    stream=True,
) as stream, open("speech.pcm", "wb") as output:
    for chunk in stream:
        delta = chunk.model_dump()["choices"][0]["delta"]
        if audio := delta.get("audio"):
            output.write(base64.b64decode(audio["data"]))
```

可用 `ffplay -f s16le -ar 24000 -ac 1 speech.pcm` 播放结果。
音频按 Fish Speech 的文本分段输出：一个分段完成语义生成及音频解码后才可发送，
短文本通常只有一段。初始 role 事件不表示音频已经生成。客户端断开后，后台会
完成当前推理和内存清理，然后才释放模型锁；优雅关闭也会等待这部分工作。

## 本地扩展与能力边界

两个接口都可通过 `fish_speech` 对象传入以下参数。OpenAI SDK 中使用
`extra_body={"fish_speech": {...}}`：

| 参数 | 默认值 |
| --- | --- |
| `max_new_tokens` | 1024；范围 1–4096，作用于每个文本分段 |
| `chunk_length` | 300；范围 100–1000 |
| `iterative_prompt` | `true` |
| `top_p` / `temperature` | 0.8；范围 0.1–1.0 |
| `repetition_penalty` | 1.1；范围 0.9–2.0 |
| `seed` | `null` |
| `use_memory_cache` | `"on"` |
| `reference_text` | 仅用于 Chat 的 Base64 参考音频 |

当前实现不支持聊天回答、user/system 提示词、音色设计、文本智能润色和
`audio.optimize_text_preview=true`。未知字段和不支持的能力返回 HTTP 400，
不会静默忽略。MiMo 的模型名称不能直接作为本地模型 ID 使用。

错误响应采用 `{"error":{"message":...,"type":...,"param":...,"code":...}}`。
输入错误为 400，密钥错误为 401，推理失败为 500，正在关闭时拒绝新推理并返回
503。响应带 `X-Request-ID`。流式请求在 HTTP 头发送之后发生错误时，会发送
包含 `error` 的 SSE 对象并以 `[DONE]` 结束，不发送成功结束标记。
