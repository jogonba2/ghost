import random
from collections import defaultdict

import numpy as np


def undersample(
    examples_by_label,
    counts_per_label,
    target_label="None",
    statistic=np.median,
):
    # If target label is not present -> don't undersample
    if target_label not in counts_per_label:
        return examples_by_label, counts_per_label
    # If there is only one class -> don't undersample
    if len(counts_per_label) == 1:
        return examples_by_label, counts_per_label

    target_num_samples = int(
        statistic(
            [
                counts_per_label[label]
                for label in counts_per_label
                if label != target_label
            ]
        )
    )
    examples_by_label[target_label] = examples_by_label[target_label][
        :target_num_samples
    ]
    counts_per_label[target_label] = len(examples_by_label[target_label])
    return examples_by_label, counts_per_label


def ensure_label_representation(examples, multilabel):
    examples_by_label = defaultdict(list)
    for example in examples:
        # Include only examples with labels
        if example["labels"]:
            if multilabel:
                examples_by_label[",".join(example["labels"])].append(example)
            else:
                examples_by_label[example["labels"][0]].append(example)
    counts_per_label = {
        label: len(label_examples)
        for label, label_examples in examples_by_label.items()
    }

    # Undersample the None class if exists
    examples_by_label, counts_per_label = undersample(
        examples_by_label, counts_per_label, target_label="None"
    )
    rng = random.Random(42)
    train_examples = []
    for label, n in counts_per_label.items():
        if n > 1024:
            train_examples.extend(rng.sample(examples_by_label[label], 512))
        elif n < 4:
            train_examples.extend(rng.choices(examples_by_label[label], k=4))
        else:
            train_examples.extend(examples_by_label[label])
    rng.shuffle(train_examples)
    return train_examples
