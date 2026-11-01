import copy
import unittest

import torch
from torch.nn import functional as F

from fish_speech.models.text2semantic.inference import decode_one_token_ar, generate
from fish_speech.models.text2semantic.llama import (
    Attention,
    BaseModelArgs,
    DualARModelArgs,
    DualARTransformer,
    KVCache,
    precompute_freqs_cis,
)
from fish_speech.models.text2semantic.mps_compile import mps_in_window, mps_sdpa


class _Tokenizer:
    eos_token_id = 15

    def get_token_id(self, _token):
        return self.eos_token_id


@unittest.skipUnless(torch.backends.mps.is_available(), "requires Apple MPS")
class MPSCompileTest(unittest.TestCase):
    def tearDown(self):
        torch._dynamo.reset()
        torch.mps.empty_cache()

    @torch.inference_mode()
    def test_native_sdpa_wrapper_preserves_outputs_for_growing_prefix(self):
        torch.manual_seed(123)
        q = torch.randn(1, 2, 1, 8, device="mps", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 8, device="mps", dtype=torch.float16)
        v = torch.randn_like(k)

        def attention(query, key, value, mask):
            return mps_sdpa(query, key, value, mask, 0.0, mask is None)

        for backend in ("aot_eager", "inductor"):
            compiled = torch.compile(
                attention, backend=backend, fullgraph=True, dynamic=True
            )
            for length in (3, 7, 16):
                with self.subTest(backend=backend, length=length):
                    mask = torch.ones(1, 1, 1, length, dtype=torch.bool, device="mps")
                    actual = compiled(q, k[:, :, :length], v[:, :, :length], mask)
                    expected = F.scaled_dot_product_attention(
                        q, k[:, :, :length], v[:, :, :length], attn_mask=mask
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                compiled(q, k, v, None),
                F.scaled_dot_product_attention(q, k, v, is_causal=True),
                rtol=0,
                atol=0,
            )

    @torch.inference_mode()
    def test_repeat_detection_remains_correct_inside_compiled_boolean_logic(self):
        def detect(history, token):
            return mps_in_window(history, token) & (token >= 4)

        compiled = torch.compile(
            detect, backend="inductor", fullgraph=True, dynamic=True
        )
        for length in (3, 10, 20):
            history = torch.arange(length, device="mps")
            for value in (0, 5, 30):
                token = torch.tensor([value], device="mps", dtype=torch.int32)
                torch.testing.assert_close(
                    compiled(history, token), detect(history, token)
                )

    @torch.inference_mode()
    def test_full_decode_with_repeat_penalty_preserves_greedy_tokens(self):
        torch.manual_seed(4)
        model = (
            DualARTransformer(
                DualARModelArgs(
                    vocab_size=16,
                    n_layer=1,
                    n_head=2,
                    n_local_heads=1,
                    dim=8,
                    intermediate_size=16,
                    head_dim=4,
                    max_seq_len=16,
                    tie_word_embeddings=False,
                    codebook_size=8,
                    num_codebooks=2,
                    semantic_begin_id=4,
                    semantic_end_id=11,
                    n_fast_layer=1,
                    fast_dim=8,
                    fast_n_head=2,
                    fast_n_local_heads=1,
                    fast_head_dim=4,
                    fast_intermediate_size=16,
                )
            )
            .eval()
            .to(device="mps", dtype=torch.float16)
        )
        model.tokenizer = _Tokenizer()
        prompt = torch.tensor([[1, 2, 3], [0, 1, 2], [3, 4, 5]], device="mps")
        kwargs = dict(
            prompt=prompt,
            max_new_tokens=4,
            audio_masks=None,
            audio_parts=None,
            temperature=0.8,
            top_p=0.8,
            top_k=1,
            repetition_penalty=1.1,
        )
        expected = generate(model=copy.deepcopy(model), **kwargs)
        for backend in ("aot_eager", "inductor"):
            compiled = torch.compile(
                decode_one_token_ar, backend=backend, fullgraph=True, dynamic=True
            )
            actual = generate(
                model=copy.deepcopy(model), decode_one_token=compiled, **kwargs
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_compiled_attention_preserves_cache_updates_and_outputs(self):
        torch.manual_seed(124)
        config = BaseModelArgs(
            dim=16,
            n_head=2,
            n_local_heads=1,
            head_dim=8,
            intermediate_size=32,
            max_seq_len=16,
            dropout=0.0,
        )
        eager = Attention(config).eval()
        eager.kv_cache = KVCache(1, 16, 1, 8, dtype=torch.float16)
        eager = eager.to(device="mps", dtype=torch.float16)
        freqs = precompute_freqs_cis(16, 8).to("mps")
        for backend in ("aot_eager", "inductor"):
            reference = copy.deepcopy(eager)
            model = copy.deepcopy(eager)
            compiled = torch.compile(
                model, backend=backend, fullgraph=True, dynamic=True
            )
            for position in range(8):
                x = torch.randn(1, 1, 16, device="mps", dtype=torch.float16)
                input_pos = torch.tensor([position], device="mps")
                mask = torch.ones(1, 1, 1, position + 1, device="mps", dtype=torch.bool)
                args = (x, freqs[position : position + 1], mask, input_pos)
                expected = reference(*args)
                actual = compiled(*args)
                torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
            torch.testing.assert_close(
                model.kv_cache.k_cache, reference.kv_cache.k_cache
            )
            torch.testing.assert_close(
                model.kv_cache.v_cache, reference.kv_cache.v_cache
            )


if __name__ == "__main__":
    unittest.main()
