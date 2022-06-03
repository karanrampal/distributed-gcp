"""Unit tests for metrics"""

import sys
sys.path.insert(0, "./src/")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

from model.net import avg_acc_gpu, avg_f1_score_gpu, confusion_matrix


def test_accuracy():
    """Test implementation of accuracy"""
    num_classes = 4
    num_examples = 3

    output = torch.randn(num_examples, num_classes)
    preds = output.argmax(dim=1)
    labels = torch.randint(0, num_classes, (num_examples,))

    sk_acc = accuracy_score(labels.numpy(), preds.numpy())
    my_acc = avg_acc_gpu(output, labels)

    assert np.isclose(sk_acc, my_acc)
