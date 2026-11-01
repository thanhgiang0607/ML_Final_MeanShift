from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        pca = PCA(n_components=2, random_state=0).fit(StandardScaler().fit_transform(data.data))
        reduced = pd.DataFrame(pca.transform(StandardScaler().fit_transform(data.data)), columns=["pc1", "pc2"])
        reduced["label"] = frame["label"]
        reduced["label_name"] = frame["label_name"]
        reduced.to_csv(OUT / f"{name}_pca2.csv", index=False)
        print(f"{path.name}: {len(frame)} dòng, {data.data.shape[1]} đặc trưng, {len(data.target_names)} lớp; "
              f"{name}_pca2.csv giữ {pca.explained_variance_ratio_.sum():.1%} phương sai")


if __name__ == "__main__":
    main()
