import unittest

from recognition.encoder import HUGGINGFACE_MODEL_IDS, SUPPORTED_MODEL_NAMES


class EncoderConfigurationTests(unittest.TestCase):
    def test_all_baseline_model_names_are_available(self) -> None:
        self.assertEqual(
            SUPPORTED_MODEL_NAMES,
            ("resnet18", "resnet50", "dinov2", "siglip"),
        )

    def test_huggingface_model_ids_are_explicit(self) -> None:
        self.assertEqual(HUGGINGFACE_MODEL_IDS["dinov2"], "facebook/dinov2-base")
        self.assertEqual(
            HUGGINGFACE_MODEL_IDS["siglip"],
            "google/siglip2-base-patch16-224",
        )


if __name__ == "__main__":
    unittest.main()
