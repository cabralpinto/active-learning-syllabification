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
    INITIAL_PERCENTAGE = 0.01

    # Load
    train_x, train_y = load(TRAIN, LEFT, RIGHT)
    test_x, test_y = load(TEST, LEFT, RIGHT)
    # Select initial training data
    initial_ids = np.random.choice(range(len(train_x)), size=int(INITIAL_PERCENTAGE * len(train_x)), replace=False)
    initial_x, initial_y = train_x[initial_ids], train_y[initial_ids]
    pool_x, pool_y = np.delete(train_x, initial_ids, axis=0), np.delete(train_y, initial_ids, axis=0)

    classifier = MLPClassifier(
        solver="adam",
        hidden_layer_sizes=((LEFT + 1 + RIGHT) * 2),
        max_iter=100,
        random_state=None,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        learner = ActiveLearner(classifier, X_training=initial_x, y_training=initial_y)
        print(learner.score(test_x, test_y))

    for i in range(QUERIES):
        index = learner.query(pool_x)[0]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            learner.teach(pool_x[index], pool_y[index])
        pool_x = np.delete(pool_x, index, axis=0)
        pool_y = np.delete(pool_y, index)
        print(learner.score(test_x, test_y))
