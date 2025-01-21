# Introduction 

This repo contains an implementation of [Ghost adaptative thresholding](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00160) adapted to work also in multilabel settings: a procedure for the selection of the optimal decision threshold for imbalanced classification. It does not require retraining machine learning models or resampling of the training data.

# Getting Started
Ghost requires python >= 3.7

To install the Ghost package, define the PIP_EXTRA_INDEX_URL environment variable:

```bash
export PIP_EXTRA_INDEX_URL='https://YOUR-PAT-TOKEN@pkgs.dev.azure.com/symanto/_packaging/Research/pypi/simple'
```

## Install with pip

```bash
pip install symanto-ghost
```

## Install from source

```bash
pip install -e .
```

# How it works
Ghost works in binary and multi-label tasks to find label thresholds on the training set that work well at inference time with test sets. Note that it is not possible to use `ghost` in zero-shot settings, since you need a classification model trained with labeled data. The following image illustrates how Ghost works for binary classification:

![Ghost pipeline](https://pubs.acs.org/cms/10.1021/acs.jcim.1c00160/asset/images/medium/ci1c00160_0002.gif)

The `ghost` package allows you to find the thresholds and to predict using that thresholds (from step 3 onwards).

## Binary tasks

```python
label_set = BinaryLabelSet(pos_label="positive", neg_label="negative")
ghost_config = GhostConfig(
    threshold_range={"start": 0.05, "stop": 0.6, "step": 0.05},
    n_subsamples=100,
    split_ratio=0.5,
    metric_fn=cohen_kappa_score,
    random_state=42,
)
# Ref labels from your training data
ref_labels = ["positive", "negative", "positive", "negative"]
# Probabilities computed by the model trained on the training data
probs = [0.8, 0.39, 0.7, 0.49]
# Instantiate BinaryGhost
ghost = BinaryGhost(label_set, ghost_config)
# Find the threshold (also internally stored in ghost to make predictions)
threshold = ghost.find_threshold(ref_labels, probs)
# Predict using the threshold
preds = ghost.predict(probs)
# Output: ["positive", "negative", "positive", "negative"]
```

## MultiLabel tasks
```python
ghost_config = GhostConfig(
    threshold_range={"start": 0.05, "stop": 0.6, "step": 0.05},
    n_subsamples=100,
    split_ratio=0.5,
    metric_fn=cohen_kappa_score,
    random_state=42,
)
ref_labels = [["joy", "fear"], ["fear"], ["joy", "sadness"], ["sadness"]]
label_set = ["joy", "fear", "sadness"]
# probs[i][j] -> probs for example i, label j in label_set
probs = [[0.8, 0.7, 0.1], [0.4, 0.8, 0.2], [0.9, 0.51, 0.3], [0.4, 0.3, 0.25]]
# Instantiate MultiLabelGhost
ghost = MultiLabelGhost(label_set, ghost_config)
# Find the threshold for each label (also internally stored in ghost to make predictions)
thresholds = ghost.find_thresholds(ref_labels, probs)
# Predict using the thresholds
preds = ghost.predict(probs)
# Output: [["joy", "fear"], ["fear"], ["joy", "sadness"], ["sadness"]]
```

# Some results

Ghost has been tested internally on two imbalanced multi-label datasets (go-emotions and reuters21758), using Symanto Brain as classification model, comparing the results with the `kneedle` algorithm and with a baseline thresholding that uses the class percentage on the training data.

|         | go-emotions | go-emotions | go-emotions  | reuters21758 | reuters21758 | reuters21758 |
|---------|-------------|-------------|--------------|--------------|--------------|--------------|
|         | f1↑          | exact match↑ | hamming loss↓ | f1↑           | exact match↑  | hamming loss↓ |
| Ghost   | 33.06       | 14.40       | 0.05         | 54.71        | 9.20         | 0.03         |
| Kneedle | 31.58       | 1.10        | 0.16         | 22.30        | 0.10         | 0.13         |
| Freq    | 10.39       | 0.0           | 0.70            | 0.04         | 0.0          | 0.89         |

The following figures show the distribution of the predicted labels by each method along with the reference ones (test set).

![GoemotionsDist](https://lh3.googleusercontent.com/drive-viewer/AFGJ81rPn6NjOErXR1GLPlU4YZ8EthIjTnccnVldw8pZrb76xpVBtsoupqkTJH6CtN-6K0o2c5_u3BYoVBiuP6WDWWggg0jT=s2560)
![Reuters21758Dist](https://lh3.googleusercontent.com/drive-viewer/AFGJ81qsgMHDAt07rEipWpd75Vw-svjWC0gJO0LjfOvqvbZ_1g9d1CcWy3pBlrPAESKIJFn3q-qIuSZfnluywQJbxgShl1bEag=s2560)




# Contribute
Please install and use symanto-dev-tools for correctly formatting the code when contributing to this repo.
