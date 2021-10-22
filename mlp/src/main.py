from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from more_itertools import windowed, flatten
import warnings


def load(filename: str, left: int, right: int):
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
    return inputs, targets


if __name__ == "__main__":
    TRAIN = "data/french/train.txt"
    TEST = "data/french/test.txt"
    LEFT = 2
    RIGHT = 2

    inputs, targets = load(TRAIN, LEFT, RIGHT)
    classifier = MLPClassifier(
        solver="adam",
        hidden_layer_sizes=((LEFT + 1 + RIGHT) * 2),
        max_iter=100,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        classifier.fit(inputs, targets)
    inputs, targets = load(TEST, LEFT, RIGHT)
    print(classifier.score(inputs, targets))
