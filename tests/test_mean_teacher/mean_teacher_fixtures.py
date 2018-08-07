import numpy as np
import pytest

from btplda.btplda.mean_teacher import MeanTeacher
from btplda.tests import plda
from btplda.tests.utils import generate_data
from numpy.testing import assert_array_equal


@pytest.fixture(scope='module')
def data_dictionary():
    np.random.seed(2345)

    n_k = 100  # Do not set this to over 100. Combinatorically worse!!!
    K = 20
    dimensionality = 10

    return generate_data(n_k, K, dimensionality)


@pytest.fixture(scope='module')
def classifier_and_latent_data(data_dictionary):
    X = data_dictionary['X']
    Y = data_dictionary['Y']

    k = np.random.choice(Y)
    X_k = X[np.asarray(Y) == k]

    classifier = plda.Classifier()
    classifier.fit_model(X, Y)
    model = classifier.model

    U_model_k = model.transform(X_k, from_space='D', to_space='U_model')

    return {'classifier': classifier, 'U_model': U_model_k, 'category': k}


@pytest.fixture(scope='module')
def teacher(classifier_and_latent_data):
    U_model = classifier_and_latent_data['U_model']
    model = classifier_and_latent_data['classifier'].model

    category_teacher = MeanTeacher(U_model, model)
    expected_shape = U_model.shape

    assert category_teacher.data.shape == expected_shape
    assert category_teacher.N == expected_shape[0]
    assert len(category_teacher.data.shape) == 2

    assert expected_shape[-1] == category_teacher.prior_cov_diag.shape[0]
    assert len(category_teacher.prior_cov_diag.shape) == 1

    expected = model.prior_params['cov_diag']
    actual = category_teacher.prior_cov_diag

    assert_array_equal(actual, expected)

    return category_teacher


@pytest.fixture(scope='module')
def test_args(classifier_and_latent_data):
    U_model = classifier_and_latent_data['U_model']
    shape = U_model.shape

    return {'N': shape[0], 'v': np.zeros(shape[1]), 'U_model': U_model}
