import unittest

import torch

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.utils.schema import ServeReferenceAudio


class ReferenceCacheTest(unittest.TestCase):
    def _make_loader(self):
        loader = ReferenceLoader()
        encoded: list[bytes] = []

        def encode_reference(reference_audio, enable_reference_audio):
            self.assertTrue(enable_reference_audio)
            encoded.append(reference_audio)
            return torch.tensor([len(encoded)], device="cpu")

        loader.encode_reference = encode_reference
        return loader, encoded

    def test_cache_off_reencodes_without_populating_cache(self) -> None:
        loader, encoded = self._make_loader()
        references = [ServeReferenceAudio(audio=b"same audio", text="first")]

        loader.load_by_hash(references, use_cache="off")
        loader.load_by_hash(references, use_cache="off")

        self.assertEqual(len(encoded), 2)
        self.assertEqual(loader.ref_by_hash, {})
        self.assertFalse(loader.last_reference_cache_stats.enabled)
        self.assertEqual(loader.last_reference_cache_stats.bypassed, 1)

    def test_cache_on_reuses_cpu_tokens_but_not_stale_text(self) -> None:
        loader, encoded = self._make_loader()
        first = [ServeReferenceAudio(audio=b"same audio", text="first")]
        corrected = [ServeReferenceAudio(audio=b"same audio", text="corrected")]

        first_tokens, first_texts = loader.load_by_hash(first, use_cache="on")
        second_tokens, second_texts = loader.load_by_hash(
            corrected, use_cache="on"
        )

        self.assertEqual(len(encoded), 1)
        self.assertEqual(first_texts, ["first"])
        self.assertEqual(second_texts, ["corrected"])
        self.assertEqual(first_tokens[0].device.type, "cpu")
        torch.testing.assert_close(first_tokens[0], second_tokens[0])
        self.assertEqual(loader.last_reference_cache_stats.hits, 1)
        self.assertEqual(loader.last_reference_cache_stats.misses, 0)


if __name__ == "__main__":
    unittest.main()
