import unittest

from src.scripts import build_feature_occurrence_index_and_labeler as labeler


class LabelQualityTests(unittest.TestCase):
    def test_low_information_filter_rejects_stopwords(self) -> None:
        self.assertTrue(labeler._is_low_information_token("the"))
        self.assertTrue(labeler._is_low_information_token("and"))
        self.assertTrue(labeler._is_low_information_token("to"))
        self.assertTrue(labeler._is_low_information_token("і"))
        self.assertTrue(labeler._is_low_information_token("та"))
        self.assertTrue(labeler._is_low_information_token("в"))
        self.assertFalse(labeler._is_low_information_token("oligopoly"))

    def test_ranking_prefers_higher_token_diversity(self) -> None:
        sampled = {
            1: [
                labeler.OccurrenceContext(context="a [the] b", token="the", token_id=1),
                labeler.OccurrenceContext(context="c [the] d", token="the", token_id=1),
                labeler.OccurrenceContext(context="e [the] f", token="the", token_id=1),
            ],
            2: [
                labeler.OccurrenceContext(context="a [neuron] b", token="neuron", token_id=2),
                labeler.OccurrenceContext(context="c [feature] d", token="feature", token_id=2),
                labeler.OccurrenceContext(context="e [semantics] f", token="semantics", token_id=2),
            ],
        }
        feature_counts = {1: 100, 2: 30}

        ranked = labeler.order_features_for_labeling(feature_counts, sampled)
        self.assertEqual(ranked[0], 2)

    def test_generic_label_detector_flags_broad_labels(self) -> None:
        self.assertTrue(labeler.is_generic_label("Key Concept Identification"))
        self.assertTrue(labeler.is_generic_label("Key Semantic Elements"))
        self.assertFalse(labeler.is_generic_label("US Federal Institutions"))

    def test_quality_gate_filters_low_entropy_features(self) -> None:
        sampled = {
            1: [
                labeler.OccurrenceContext(context="a [the] b", token="the", token_id=1),
                labeler.OccurrenceContext(context="c [the] d", token="the", token_id=1),
                labeler.OccurrenceContext(context="e [the] f", token="the", token_id=1),
            ],
            2: [
                labeler.OccurrenceContext(context="a [neuron] b", token="neuron", token_id=2),
                labeler.OccurrenceContext(context="c [feature] d", token="feature", token_id=2),
                labeler.OccurrenceContext(context="e [semantics] f", token="semantics", token_id=2),
            ],
        }
        feature_counts = {1: 100, 2: 30}
        ranked = labeler.order_features_for_labeling(feature_counts, sampled)

        gated = labeler.apply_feature_quality_gate(
            ranked,
            sampled,
            min_unique_tokens=2,
            min_diversity_ratio=0.5,
            min_token_entropy=0.7,
        )
        self.assertEqual(gated, [2])

    def test_context_window_glues_fragmented_tokens(self) -> None:
        tokens = [
            {"token_str": "Таким", "token_id": 1},
            {"token_str": " чи", "token_id": 2},
            {"token_str": " ном", "token_id": 3},
            {"token_str": ",", "token_id": 4},
            {"token_str": " зви", "token_id": 5},
            {"token_str": " ну", "token_id": 6},
            {"token_str": " ва", "token_id": 7},
            {"token_str": " чення", "token_id": 8},
        ]
        context, fired_token, _ = labeler.build_context_window(tokens, fired_token_index=5, radius=4)
        self.assertEqual(fired_token, "ну")
        self.assertIn("чи ном, зви [ну] ва чення", context)


if __name__ == "__main__":
    unittest.main()
