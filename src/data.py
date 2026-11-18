"""Loads the real Fashion-MNIST dataset (60,000 train / 10,000 test grayscale
28x28 product photos across 10 real apparel categories) from the official
IDX files. This is Zalando's actual benchmark release, not a synthetic or
toy dataset - a believable stand-in for an e-commerce catalog image
classifier, which is the real use case this project targets: can an
attacker perturb a product photo to get it miscategorized, and can the
classifier be made robust to that."""
import gzip
import struct

import numpy as np

ROOT = "data"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _read_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        buf = f.read(n * rows * cols)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        buf = f.read(n)
        return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_fashion_mnist():
    Xtr = _read_images(f"{ROOT}/train-images-idx3-ubyte.gz")
    ytr = _read_labels(f"{ROOT}/train-labels-idx1-ubyte.gz")
    Xte = _read_images(f"{ROOT}/t10k-images-idx3-ubyte.gz")
    yte = _read_labels(f"{ROOT}/t10k-labels-idx1-ubyte.gz")
    return Xtr[:, None, :, :], ytr, Xte[:, None, :, :], yte  # add channel dim
