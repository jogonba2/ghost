import unittest

from ghost.ghost import BinaryGhost, MultiLabelGhost
from ghost.models import BinaryLabelSet, GhostConfig
from sklearn.metrics import cohen_kappa_score


class TestBinaryGhost(unittest.TestCase):
    def test_binary_ghost(self):
        label_set = BinaryLabelSet(pos_label="positive", neg_label="negative")
        ghost_config = GhostConfig(
            threshold_range={"start": 0.05, "stop": 0.6, "step": 0.05},
            n_subsamples=100,
            split_ratio=0.5,
            metric_fn=cohen_kappa_score,
            random_state=42,
        )
        ref_labels = ["positive", "negative", "positive", "negative"]
        probs = [0.8, 0.39, 0.7, 0.49]
        ghost = BinaryGhost(label_set, ghost_config)
        threshold = ghost.find_threshold(ref_labels, probs)
        self.assertEqual(threshold, 0.5)
        preds = ghost.predict(probs)
        self.assertListEqual(preds, ref_labels)


class TestMultiLabelGhost(unittest.TestCase):
    def test_multilabel_ghost(self):
        ghost_config = GhostConfig(
            threshold_range={"start": 0.05, "stop": 0.6, "step": 0.05},
            n_subsamples=100,
            split_ratio=0.5,
            metric_fn=cohen_kappa_score,
            random_state=42,
        )
        ref_labels = [["joy", "fear"], ["fear"], ["joy", "sadness"], ["sadness"]]
        label_set = ["joy", "fear", "sadness"]
        probs = [[0.8, 0.7, 0.1], [0.4, 0.8, 0.2], [0.9, 0.51, 0.3], [0.4, 0.3, 0.25]]
        ghost = MultiLabelGhost(label_set, ghost_config)
        ghost.find_thresholds(ref_labels, probs)
        preds = ghost.predict(probs)
        self.assertListEqual(preds, ref_labels)


if __name__ == "__main__":
    unittest.main()
