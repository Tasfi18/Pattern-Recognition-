import numpy as np
import pandas as pd

import naive_bayes as nb
from forgy_kmeans import WCSS, forgy_algorithm, k_means_algorithm, load_data
from knn import KNN, load_iris
from parzen import make_data, parzen_density


# ---------- Naive Bayes ----------

def test_load_csv_skips_the_id_column():
    data = nb.load_csv(nb.HERE / "iris.csv")
    features, label = data[0]
    assert features == [5.1, 3.5, 1.4, 0.2]
    assert label == "Iris-setosa"
    assert len(data) == 150


def test_accuracy_counts_each_correct_prediction_once(capsys):
    train = [([1.0], "a"), ([1.2], "a"), ([0.8], "a"), ([5.0], "b"), ([5.3], "b"), ([4.8], "b")]
    test = [([1.1], "a"), ([5.1], "b")]
    summary = nb.information_classes(train)
    priors = nb.prior_probabilities(train)
    assert nb.test_model(summary, test, priors) == 100.0  # was 200% before the fix


def test_probabilities_sum_to_one_and_survive_underflow():
    summary = {"a": [(0.0, 1.0)], "b": [(10.0, 1.0)]}
    priors = {"a": 0.5, "b": 0.5}
    probs = nb.class_probability(summary, [2.0], priors)
    assert abs(sum(probs.values()) - 1) < 1e-9
    assert nb.class_probability(summary, [1e6], priors) == priors  # both likelihoods are 0


def test_naive_bayes_classifies_iris_well():
    import random
    random.seed(1)
    data = nb.load_csv(nb.HERE / "iris.csv")
    train, test = nb.split_data(data, 0.7)
    summary = nb.information_classes(train)
    priors = nb.prior_probabilities(train)
    correct = sum(nb.predict(summary, row, priors)[0] == label for row, label in test)
    assert correct / len(test) > 0.85


# ---------- KNN ----------

def test_knn_uses_the_majority_of_the_nearest_neighbours():
    model = KNN(3)
    model.train([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11]], ["a", "a", "a", "b", "b"])
    assert model.predict([[0.5, 0.5], [9, 9]]) == ["a", "b"]


def test_knn_on_iris():
    data = load_iris()
    X = data.drop("Species", axis=1).to_numpy()
    y = data["Species"].to_numpy()
    model = KNN(5)
    model.train(X[::2], y[::2])
    predictions = model.predict(X[1::2])
    assert np.mean(np.array(predictions) == y[1::2]) > 0.9


# ---------- Parzen window ----------

def test_parzen_density_integrates_to_about_one():
    data = make_data(n=400)
    grid = np.linspace(-6, 11, 2000)
    density = parzen_density(data, grid, h=0.5)
    area = np.sum(density) * (grid[1] - grid[0])
    assert abs(area - 1) < 0.01


def test_parzen_density_peaks_near_the_two_means():
    data = make_data(n=1000)
    grid = np.linspace(-3, 8, 1101)
    density = parzen_density(data, grid, h=0.5)
    assert density[np.argmin(abs(grid - 0))] > density[np.argmin(abs(grid - 2.5))]
    assert density[np.argmin(abs(grid - 5))] > density[np.argmin(abs(grid - 2.5))]


# ---------- Clustering ----------

def test_more_clusters_never_fit_worse():
    df = load_data()
    w2, w4 = forgy_algorithm(df, [2, 4], plot=False)
    assert w4 < w2
    k2, k4 = k_means_algorithm(df, [2, 4], plot=False)
    assert k4 < k2


def test_forgy_finds_two_obvious_groups():
    df = pd.DataFrame([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
    (wcss,) = forgy_algorithm(df, [2], plot=False)
    assert wcss < 3  # each point is within about 0.75 of its group's centre


def test_wcss_adds_squared_distances():
    assert WCSS([(1.0, None), (2.0, None)]) == 5.0
