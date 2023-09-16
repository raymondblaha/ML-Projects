from sklearn.datasets import fetch_openml
from os import path
import pickle
import ssl

def load_data():
    cache_path = "cache_mnist.pkl"
    if path.exists(cache_path):
        with open(cache_path, "rb") as f:
            X = pickle.load(f)
            y = pickle.load(f)
        print("Read data from cache.")
    else:
        # This might take a while, so let the user know what is
        # going on
        print("Fetching...", end="", flush=True)

        # Accept unverified certificates for fetch
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Fetch the data
        mnist = fetch_openml("mnist_784", as_frame=False, parser="auto")
        X = mnist.data
        y = mnist.target
        print("Done.")

        # Cache in a file
        with open(cache_path, "wb") as f:
            pickle.dump(X, f)
            pickle.dump(y, f)
    return X, y
