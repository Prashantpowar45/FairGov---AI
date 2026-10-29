from .generate import make_dataset
from .validate import validate
from .train import train

if __name__ == "__main__":
    print("Generating new DTE synthetic dataset...")
    make_dataset()
    print("Validation:", validate())
    print("Training/evaluation:", train())
