import warnings

import numpy as np
from modAL.models import ActiveLearner
from more_itertools import windowed
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


def load(filename: str, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    with open(filename, encoding="utf-8") as file:
        words = file.read().split("\n\n")[:-1]
    inputs = [
        window
        for word in words
        for window in windowed(
            left * [-1] + [ord(char) for char in word[0::4]] + right * [-1],
            left + 1 + right,
        )
    ]
    targets = [int(char) for word in words for char in word[2::4]]
    return np.array(inputs), np.array(targets)


if __name__ == "__main__":
    TRAIN = "data/french/train.txt"
    TEST = "data/french/test.txt"
    LEFT = 2
    RIGHT = 2
    QUERIES = 1000

    train_inputs, train_targets = load(TRAIN, LEFT, RIGHT)
    test_inputs, test_targets = load(TEST, LEFT, RIGHT)
    classifier = MLPClassifier(
        solver="adam",
        hidden_layer_sizes=((LEFT + 1 + RIGHT) * 2),
        max_iter=100,
        random_state=None,
    )
    learner = ActiveLearner(classifier)
    print(len(train_inputs))
    for i in range(QUERIES):
        print(train_inputs[0])
        index = learner.query(train_inputs)[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            learner.teach(train_inputs[index], train_targets[index])
        train_inputs = np.delete(train_inputs, index, axis=0)
        train_targets = np.delete(train_targets, index)
        print(learner.score(test_inputs, test_targets))
