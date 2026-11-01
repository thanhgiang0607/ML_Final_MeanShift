from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

OUT = Path(__file__).resolve().parent
LOADERS = {"iris": load_iris, "wine": load_wine, "breast_cancer": load_breast_cancer}


def main():
    for name, loader in LOADERS.items():
        data = loader()
        frame = pd.DataFrame(data.data, columns=data.feature_names)
        frame["label"] = data.target
        frame["label_name"] = [data.target_names[t] for t in data.target]
        path = OUT / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"{path.name}: {len(frame)} dòng, {data.data.shape[1]} đặc trưng, {len(data.target_names)} lớp")


if __name__ == "__main__":
    main()
