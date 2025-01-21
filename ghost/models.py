from typing import Callable, List

from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score


class BinaryLabelSet(BaseModel):
    pos_label: str
    neg_label: str


class MultiLabelSet(BaseModel):
    labels: List[str]


class ThresholdRange(BaseModel):
    start: float = 0.05
    stop: float = 0.6
    step: float = 0.05


class GhostConfig(BaseModel):
    threshold_range: ThresholdRange
    n_subsamples: int = 100
    split_ratio: float = 0.5
    metric_fn: Callable = cohen_kappa_score
    random_state: int = 42
