from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from ghost.models import BinaryLabelSet, GhostConfig, MultiLabelSet


class MultiLabelGhost:
    def __init__(self, label_set: MultiLabelSet, ghost_config: GhostConfig):
        self.ghost_config = ghost_config
        self.label_set = label_set
        self.best_thresholds: Dict[str, float] = {}

    def decision_rule(
        self, label: str, prob: float, threshold: float
    ) -> Optional[str]:
        if prob >= threshold:
            return label
        return None

    def find_thresholds(
        self, ref_labels: List[List[str]], probs: List[List[float]]
    ) -> Dict[str, float]:
        thresholds = {}
        for label in self.label_set:
            binary_label_set = BinaryLabelSet(
                pos_label=label, neg_label=f"no_{label}"
            )
            ref_label = []
            label_probs = []
            for ex_labels, ex_probs in zip(ref_labels, probs):
                if label in ex_labels:
                    ref_label.append(label)
                else:
                    ref_label.append(binary_label_set.neg_label)
                label_probs.append(ex_probs[self.label_set.index(label)])
            binary_ghost = BinaryGhost(
                label_set=binary_label_set, ghost_config=self.ghost_config
            )
            threshold = binary_ghost.find_threshold(
                ref_label,
                label_probs,
            )
            thresholds[label] = threshold
        self.best_thresholds = thresholds
        return self.best_thresholds

    def predict(self, probs: List[List[float]]) -> List[List[str]]:
        assert (
            self.best_thresholds is not None
        ), "You should find the thresholds before predicting"
        preds = []
        for ex_probs in probs:
            ex_preds = []
            for label, prob in zip(self.label_set, ex_probs):
                pred_label = self.decision_rule(
                    label, prob, self.best_thresholds[label]
                )
                if pred_label is not None:
                    ex_preds.append(pred_label)
            preds.append(ex_preds)
        return preds


class BinaryGhost:
    def __init__(self, label_set: BinaryLabelSet, ghost_config: GhostConfig):
        self.ghost_config = ghost_config
        self.label_set = label_set
        self.best_threshold = -1.0

    def decision_rule(self, prob: float, threshold: float) -> str:
        if prob >= threshold:
            return self.label_set.pos_label
        return self.label_set.neg_label

    def loop_thresholds(
        self,
        probs: List[float],
    ) -> Dict[float, List[str]]:
        threshold_preds = {}
        for threshold in np.arange(
            self.ghost_config.threshold_range.start,
            self.ghost_config.threshold_range.stop,
            self.ghost_config.threshold_range.step,
        ):
            threshold_preds[threshold] = np.array(
                [self.decision_rule(prob, threshold) for prob in probs]
            )
        return threshold_preds

    def find_threshold(
        self,
        ref_labels: List[str],
        probs: List[float],
    ) -> float:
        # Use a range of thresholds to label the training samples
        threshold_preds = self.loop_thresholds(probs)
        # N random stratified subsamples are drawn from the training set
        sampling = StratifiedShuffleSplit(
            n_splits=self.ghost_config.n_subsamples,
            test_size=1.0 - self.ghost_config.split_ratio,
            random_state=self.ghost_config.random_state,
        )
        splits = sampling.split(X=range(len(probs)), y=ref_labels)
        # The evaluation metric is calculated for every subset and for each of the thresholds
        split_eval = self._eval_splits(splits, ref_labels, threshold_preds)
        # Pick threshold that maximizes the median metric
        self.best_threshold = pd.DataFrame(split_eval).median(axis=1).idxmax()
        return self.best_threshold

    def _eval_splits(
        self,
        splits,
        ref_labels: List[str],
        threshold_preds: Dict[float, List[str]],
    ) -> Dict[int, Dict[float, float]]:
        split_eval: Dict[int, Dict[float, float]] = {}
        for i, (split_idxs, _) in enumerate(splits):
            split_ref_labels = np.array(ref_labels)[split_idxs]
            split_eval[i] = {}
            for threshold in threshold_preds:
                split_pred_labels = np.array(threshold_preds[threshold])[
                    split_idxs
                ]
                split_eval[i][threshold] = self.ghost_config.metric_fn(
                    split_ref_labels, split_pred_labels
                )
        return split_eval

    def predict(self, probs: List[float]) -> List[str]:
        assert (
            self.best_threshold is not None
        ), "You should find the threshold before predicting"
        return [self.decision_rule(prob, self.best_threshold) for prob in probs]
