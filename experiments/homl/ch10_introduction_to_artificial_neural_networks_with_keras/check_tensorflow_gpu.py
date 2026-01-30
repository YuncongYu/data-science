import sys

import pandas as pd
import sklearn as sk
import tensorflow as tf
import tensorflow.keras


def main():
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tensorflow.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")


if __name__ == "__main__":
    main()
