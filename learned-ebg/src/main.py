from sklearn.neural_network import MLPClassifier
from more_itertools import windowed


def load(filename: str, left: int, right: int):
    with open(filename) as file:
        words = [
            [phone.split("\t") for phone in word.split("\n")]
            for word in file.read().split("\n\n")[:-1]
        ]
    inputs, targets = zip(*(zip(*word) for word in words))
    inputs = [
        list(windowed(left * ("",) + input + right * ("",), left + 1 + right))
        for input in inputs
    ]
    return inputs, targets


if __name__ == "__main__":
    TRAIN = "data/french/small.txt"
    TEST = "data/french/test.txt"
    LEFT = 2
    RIGHT = 2
    print(*load(TRAIN, LEFT, RIGHT), sep="\n\n")

# classifier = MLPClassifier(
#     solver="lbfgs",
#     alpha=1e-5,
#     hidden_layer_sizes=(5, 2),
#     random_state=1,
# )
