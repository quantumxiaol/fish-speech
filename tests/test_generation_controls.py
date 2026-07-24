import queue
import unittest
from types import SimpleNamespace

import torch

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import (
    RAS_WIN_SIZE,
    decode_n_tokens,
    ensure_model_caches,
    logits_to_probs,
    prepare_text_batches,
)
from fish_speech.utils.schema import ServeTTSRequest
from tools.fastapi_service import _build_tts_request
from tools.fish_client import _collect_generation_kwargs, build_parser


class RepetitionPenaltyTest(unittest.TestCase):
    def test_model_cache_storage_is_initialized_once(self) -> None:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(max_seq_len=4096)
                self.setup_calls = 0

            def setup_caches(self, **_kwargs) -> None:
                self.setup_calls += 1

        model = FakeModel()
        ensure_model_caches(model, device="cpu")
        ensure_model_caches(model, device="cpu")

        self.assertEqual(model.setup_calls, 1)
        self.assertTrue(model._cache_setup_done)

    def test_repetition_penalty_changes_seen_token_scores(self) -> None:
        logits = torch.tensor([2.0, -2.0, 1.0])
        original = logits.clone()

        probs = logits_to_probs(
            logits,
            temperature=torch.tensor(1.0),
            top_p=torch.tensor(1.0),
            top_k=3,
            repetition_penalty=torch.tensor(2.0),
            previous_tokens=torch.tensor([0, 1]),
        )

        expected = torch.softmax(torch.tensor([1.0, -4.0, 1.0]), dim=-1)
        torch.testing.assert_close(probs, expected)
        torch.testing.assert_close(logits, original)

    def test_decode_loop_forwards_penalty_and_uses_fixed_history(self) -> None:
        calls: list[dict] = []

        class Tokenizer:
            @staticmethod
            def get_token_id(_token: str) -> int:
                return 999

        model = SimpleNamespace(
            config=SimpleNamespace(num_codebooks=1),
            tokenizer=Tokenizer(),
        )
        repetition_penalty = torch.tensor(1.25)

        def fake_decode_one_token(**kwargs):
            recorded = kwargs.copy()
            recorded["previous_tokens"] = kwargs["previous_tokens"].clone()
            calls.append(recorded)
            step = len(calls)
            return torch.tensor([[10 + step], [20 + step]], dtype=torch.int)

        output = decode_n_tokens(
            model=model,
            cur_token=torch.tensor([[[10]], [[20]]], dtype=torch.int),
            input_pos=torch.tensor([1]),
            num_new_tokens=2,
            temperature=torch.tensor(0.8),
            top_p=torch.tensor(0.9),
            top_k=30,
            repetition_penalty=repetition_penalty,
            semantic_logit_bias=torch.zeros(1),
            audio_masks=None,
            audio_parts=None,
            decode_one_token=fake_decode_one_token,
        )

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["repetition_penalty"], repetition_penalty)
        self.assertEqual(
            tuple(calls[0]["previous_tokens"].shape),
            (2, RAS_WIN_SIZE),
        )
        self.assertTrue((calls[0]["previous_tokens"][0] == 10).all())
        self.assertTrue((calls[0]["previous_tokens"][1] == 20).all())


class IterativePromptTest(unittest.TestCase):
    def test_disabled_iterative_prompt_keeps_text_in_one_batch(self) -> None:
        text = "第一句。第二句。第三句。"
        self.assertEqual(
            prepare_text_batches(
                text,
                iterative_prompt=False,
                chunk_length=12,
            ),
            [text],
        )

    def test_plain_text_is_split_on_utf8_safe_boundaries(self) -> None:
        text = "第一句。第二句很长，第三句结束。"
        batches = prepare_text_batches(
            text,
            iterative_prompt=True,
            chunk_length=18,
        )

        self.assertGreater(len(batches), 1)
        self.assertEqual("".join(batches), text)
        self.assertTrue(
            all(len(batch.encode("utf-8")) <= 18 for batch in batches)
        )

    def test_engine_forwards_api_generation_controls(self) -> None:
        engine = object.__new__(TTSInferenceEngine)
        engine.decoder_model = SimpleNamespace(device=torch.device("cpu"))
        engine.compile = False
        engine.llama_queue = queue.Queue()

        request = ServeTTSRequest(
            text="test",
            iterative_prompt=False,
            repetition_penalty=1.35,
        )
        engine.send_Llama_request(request, prompt_tokens=[], prompt_texts=[])
        queued = engine.llama_queue.get_nowait()

        self.assertFalse(queued.request["iterative_prompt"])
        self.assertEqual(queued.request["repetition_penalty"], 1.35)

    def test_fastapi_form_builder_forwards_generation_controls(self) -> None:
        request = _build_tts_request(
            text="test",
            reference_audio=b"audio",
            reference_text="reference",
            audio_format="wav",
            max_new_tokens=128,
            chunk_length=300,
            iterative_prompt=False,
            top_p=0.8,
            repetition_penalty=1.4,
            temperature=0.8,
            seed=None,
            use_memory_cache="off",
        )

        self.assertFalse(request.iterative_prompt)
        self.assertEqual(request.repetition_penalty, 1.4)

    def test_http_cli_can_disable_iterative_prompt(self) -> None:
        args = build_parser().parse_args(
            [
                "voice-clone",
                "--ref-audio",
                "reference.wav",
                "--no-iterative-prompt",
                "--repetition-penalty",
                "1.4",
            ]
        )

        generation_kwargs = _collect_generation_kwargs(args)
        self.assertFalse(generation_kwargs["iterative_prompt"])
        self.assertEqual(generation_kwargs["repetition_penalty"], 1.4)


if __name__ == "__main__":
    unittest.main()
