"""Run R-GGPI feature importance and feature selection on synthetic data."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from r_ggpi import RGGPIExplainer, RGGPISelector


def main():
    X_values, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=4,
        n_redundant=3,
        random_state=42,
    )
    feature_names = [f"feature_{index}" for index in range(X_values.shape[1])]
    X = pd.DataFrame(X_values, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    result = RGGPIExplainer(model, n_repeats=3).explain(X_test, y_test)
    print("Feature importance (highest first):")
    print(result.importances_mean.sort_values(ascending=False).head())

    selector = RGGPISelector(model, n_repeats=3, max_features=5)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    print("\nSelected features:")
    print(selector.selected_features_)

    selected_model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    selected_model.fit(X_train_selected, y_train)
    print(f"Selected-feature test accuracy: {selected_model.score(X_test_selected, y_test):.3f}")


if __name__ == "__main__":
    main()
