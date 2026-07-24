import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from fish_speech.models.text2semantic.llama import (
    Attention,
    BaseModelArgs,
    KVCache,
    precompute_freqs_cis,
)


class ActiveKVAttentionTest(unittest.TestCase):
    @staticmethod
    def _attention() -> Attention:
        config = BaseModelArgs(
            vocab_size=32,
            n_layer=1,
            n_head=2,
            n_local_heads=1,
            dim=8,
            intermediate_size=16,
            head_dim=4,
            max_seq_len=8,
            dropout=0.0,
        )
        attention = Attention(config).eval()
        attention.kv_cache = KVCache(
            max_batch_size=1,
            max_seq_len=8,
            n_heads=1,
            head_dim=4,
            dtype=torch.float32,
        )
        return attention

    @staticmethod
    def _seed_cache(attention: Attention) -> None:
        torch.manual_seed(100)
        attention.kv_cache.k_cache[:, :, :2] = torch.randn(1, 1, 2, 4)
        attention.kv_cache.v_cache[:, :, :2] = torch.randn(1, 1, 2, 4)

    def test_attention_passes_only_active_kv_prefix_to_sdpa(self) -> None:
        torch.manual_seed(1)
        attention = self._attention()
        self._seed_cache(attention)
        x = torch.randn(1, 1, 8)
        freqs_cis = precompute_freqs_cis(8, 4)[2:3]
        mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        original_sdpa = F.scaled_dot_product_attention
        observed_shapes = []

        def capture_sdpa(query, key, value, **kwargs):
            observed_shapes.append((key.shape, value.shape))
            return original_sdpa(query, key, value, **kwargs)

        with patch(
            "fish_speech.models.text2semantic.llama.F.scaled_dot_product_attention",
            side_effect=capture_sdpa,
        ):
            attention(
                x,
                freqs_cis,
                mask,
                input_pos=torch.tensor([2]),
            )

        self.assertEqual(
            observed_shapes,
            [(torch.Size([1, 2, 3, 4]), torch.Size([1, 2, 3, 4]))],
        )
        self.assertEqual(attention.kv_cache.k_cache.shape[2], 8)

    def test_active_prefix_matches_full_cache_with_masked_tail(self) -> None:
        torch.manual_seed(2)
        active_attention = self._attention()
        self._seed_cache(active_attention)
        full_attention = copy.deepcopy(active_attention)
        x = torch.randn(1, 1, 8)
        freqs_cis = precompute_freqs_cis(8, 4)[2:3]
        active_mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        full_mask = torch.zeros(1, 1, 1, 8, dtype=torch.bool)
        full_mask[..., :3] = True
        input_pos = torch.tensor([2])

        active_output = active_attention(
            x,
            freqs_cis,
            active_mask,
            input_pos=input_pos,
        )
        full_output = full_attention(
            x,
            freqs_cis,
            full_mask,
            input_pos=input_pos,
        )

        torch.testing.assert_close(active_output, full_output, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
