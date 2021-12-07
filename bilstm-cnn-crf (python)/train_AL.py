# This script trains the BiLSTM-CRF architecture for syllabification.

import argparse
import logging
import os
import sys

import numpy as np
from modAL.models import ActiveLearner

from config import get_cfg_defaults
from neuralnets.BiLSTM import BiLSTM
from preprocessing import load_dataset

# Deactivate weird RuntimeWarning
np.seterr(invalid='ignore')

# Change into the working dir of the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Logging level
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Results directories
PATH = os.getcwd() + "/results/"

def create_directory(name):
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    if not os.path.exists(PATH + "/" + str(name)):
        os.mkdir(PATH + "/" + str(name))


def train_and_eval_model(cfg):
    """
    Load data and train model
    args:
        cfg (YACS YAML config)
    """

    # Data preprocessing
    dataset = {
        "columns": {0: "raw_tokens", 1: "boundaries"},
        # CoNLL format (tab-delineated)
        #   Column 0: phones
        #   Column 1: syllable boundary
        "label": "boundaries",  # Which column we like to predict
    }

    # Load the embeddings and the dataset. Choose whether or not to pad the words.
    # Right now, padding must be done if CRF is chosen for output layer.
    # The CRF layer does not support masking.
    embeddings, data, mappings, vocab_size, n_class_labels, word_length = load_dataset(
        dataset, dataset_name=cfg.TRAINING.DATASET, do_pad_words=True
    )
    create_directory(cfg.CONFIG_NAME)
    logger.info(f"Starting training of `{cfg.CONFIG_NAME}` on dataset `{dataset}`")

    QUERIES = 20
    INITIAL_PERCENTAGE = 0.01

    # Parse train and test data
    train_x = np.array([np.array(i["tokens"]) for i in data["train_matrix"]])
    train_y = np.array([np.array(i["boundaries"]) for i in data["train_matrix"]])
    test_x = np.array([np.array(i["tokens"]) for i in data["test_matrix"]])
    test_y = np.array([np.array(i["boundaries"]) for i in data["test_matrix"]])

    # Select initial training data
    initial_ids = np.random.choice(range(len(train_x)), size=int(INITIAL_PERCENTAGE * len(train_x)), replace=False)
    initial_x, initial_y = train_x[initial_ids], train_y[initial_ids]
    pool_x, pool_y = np.delete(train_x, initial_ids, axis=0), np.delete(train_y, initial_ids, axis=0)

    model = BiLSTM(cfg)
    model.set_vocab(vocab_size, n_class_labels, word_length, mappings)
    model.set_dataset(dataset, data)
    learner = ActiveLearner(model, X_training=initial_x, y_training=initial_y)
    word_level_acc, boundary_level_acc = learner.score(test_x, test_y)
    print(f"Score on query 0: Word level accuracy {word_level_acc:.5f}, Boundary level accuracy {boundary_level_acc:.5f}")
    model.save_model(0, word_level_acc, boundary_level_acc)

    for query in range(1, QUERIES + 1):
        index = learner.query(pool_x)[0][0]
        learner.teach(np.array([pool_x[index]]), np.array([pool_y[index]]))
        pool_x = np.delete(pool_x, index, axis=0)
        pool_y = np.delete(pool_y, index, axis=0)
        word_level_acc, boundary_level_acc = learner.score(test_x, test_y)
        print(f"Score on query {query}: Word level accuracy {word_level_acc:.5f}, Boundary level accuracy {boundary_level_acc:.5f}")
        model.save_model(query, word_level_acc, boundary_level_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/french_large_base.yaml",
        help="filename of config to run experiment with",
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    logging.info(cfg)
    train_and_eval_model(cfg)
