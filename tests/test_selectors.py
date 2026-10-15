import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from r_ggpi import RGGPISelector


def test_selector_fit_and_transform_with_dataframe():
  X = pd.DataFrame(
      {
          "signal": [0, 1, 0, 1, 0, 1],
          "noise": [1, 0, 1, 1, 0, 0],
      }
  )
  y = np.array([0, 1, 0, 1, 0, 1])
  model = DummyClassifier(strategy="most_frequent").fit(X, y)
  selector = RGGPISelector(model, n_repeats=2, max_features=2)

  transformed = selector.fit_transform(X, y)

  assert transformed.shape == (len(X), 2)
  assert list(transformed.columns) == selector.selected_features_