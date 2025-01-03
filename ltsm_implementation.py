import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


def main():
    with open('data/train_embeddings.pkl', 'rb') as f:
        data = joblib.load(f)
    print(data)
    

if __name__ == "__main__":
    main()