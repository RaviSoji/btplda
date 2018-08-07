import itertools
import numpy as np
import pytest

from btplda.btplda import mean_teacher
from itertools import combinations
from scipy.special import comb, logsumexp
from scipy.stats import multivariate_normal as gaussian

from .mean_teacher_fixtures import teacher, test_args
from .mean_teacher_fixtures import classifier_and_latent_data
from .mean_teacher_fixtures import data_dictionary


def test_calc_n_batches():
    N = 10
    set_size = 2
    args = (N, set_size)
    
    assert mean_teacher.calc_n_batches(*args, None, None)[0] == 1
    assert mean_teacher.calc_n_batches(*args, None, 5)[0] == 9
    assert mean_teacher.calc_n_batches(*args, 35, None)[0] == 1
    assert mean_teacher.calc_n_batches(*args, 55, None)[0] == 1
    assert mean_teacher.calc_n_batches(*args, 10, 5)[0] == 2
    assert mean_teacher.calc_n_batches(*args, 11, 5)[0] == 3
    assert mean_teacher.calc_n_batches(*args, 9, 5)[0] == 2
    assert mean_teacher.calc_n_batches(*args, 4, 5)[0] == 1


def test_generate_batches():
    def test_raises_error(batch_size=5, n_batches=50):
        N = 10
        set_size = 2
        sets = combinations(range(N), set_size)
        total_sets = comb(N, set_size)

        args = (sets, batch_size, n_batches, total_sets)
        with pytest.raises(Exception):
            next(mean_teacher.generate_batches(*args))
        
    def test_shape(N, set_size, batch_size=5, n_batches=50, expected=None):
        sets = combinations(range(N), set_size)
        total_sets = comb(N, set_size)

        args = (sets, batch_size, n_batches, total_sets)
        batches = mean_teacher.generate_batches(*args)

        for i, batch in enumerate(batches):
            arr = np.asarray(list(batch))

            assert arr.shape[-2] <= expected[-2]
            assert arr.shape[-1] == expected[-1]

        assert i + 1 == expected[-3]

    test_raises_error(batch_size=None)
    test_raises_error(n_batches=None)

    N = 10
    set_size = 2
    tot = int(comb(N, set_size))

    test_shape(N, set_size, batch_size=tot - 1,
               expected=(2, tot - 1, set_size))
    test_shape(N, set_size, batch_size=tot + 1,
               expected=(1, tot + 1, set_size))

    set_size = 1
    tot = int(comb(N, set_size))
    test_shape(N, set_size, batch_size=tot - 1,
               expected=(2, tot - 1, set_size))
    test_shape(N, set_size, batch_size=tot + 1,
               expected=(1, tot + 1, set_size))

# def test_generate_teaching_sets(teacher, test_args):
#     np.random.seed(1234)
# 
#     def test_for_set_size(sz, teacher, test_args):
#         N = test_args['N']
#         v = test_args['v']
# 
#         # Should raise error when sequential_sampling is an invalid value.
#         with pytest.raises(Exception):
#             teacher.generate_teaching_sets(v, sz,
#                                            sequential_sampling='something')
# 
#         # Test variable type and combinometrics.
#         ts = teacher.generate_teaching_sets(v, sz)
# 
#         assert type(ts) == itertools.combinations
#         assert len(list(ts)) == comb(N, sz)
#     
#         ts = teacher.generate_teaching_sets(v, sz, 'ascending')
# 
#         assert type(ts) == itertools.combinations
#         assert len(list(ts)) == comb(N, sz)
#     
#         ts = teacher.generate_teaching_sets(v, sz, 'descending')
# 
#         assert type(ts) == itertools.combinations
#         assert len(list(ts)) == comb(N, sz)
#     
#         # Test randomization (sequential_sampling=None)
#         ts_1 = list(teacher.generate_teaching_sets(v, sz))
#         ts_2 = list(teacher.generate_teaching_sets(v, sz))
# 
#         assert not np.array_equal(np.asarray(ts_1), np.asarray(ts_2))
# 
#     for set_size in [1, 2, 3]:
#         test_for_set_size(set_size, teacher, test_args)
#     
#     # sequential_sampling='ascending' vs. sequential_sampling='descending'.
#     v = test_args['v']
#     ts_1 = list(teacher.generate_teaching_sets(v, 1, 'ascending'))
#     ts_2 = list(teacher.generate_teaching_sets(v, 1, 'descending'))
# 
#     assert np.array_equal(ts_1, ts_2[::-1])


def test_calc_logp_likelihood(teacher, test_args):
    np.random.seed(1234)

    v = test_args['v']

    # Posterior for the multivariate conjugate Gaussian.
    set_size = 2
    sets = np.asarray(list(teacher.generate_teaching_sets(v, set_size)))

    prior_cov_diag = teacher.prior_cov_diag
    posterior_cov_diag = prior_cov_diag / (set_size * prior_cov_diag + 1)

    posterior_means = teacher.data[sets].mean(axis=-2)
    posterior_means = set_size * posterior_cov_diag * posterior_means
    posterior_cov = np.diag(posterior_cov_diag)

    actual = teacher.calc_logp_likelihood(v, sets)

    for i, mean in enumerate(posterior_means):
        expected = gaussian(mean, posterior_cov).logpdf(v)

        assert actual[i] == expected


    # Test that code doesn't break when teaching set size is 1.
    set_size = 1
    sets = np.asarray(list(teacher.generate_teaching_sets(v, set_size)))


    prior_cov_diag = teacher.prior_cov_diag
    posterior_cov_diag = prior_cov_diag / (set_size * prior_cov_diag + 1)

    posterior_means = np.squeeze(teacher.data[sets])  # since set_size = 1.
    posterior_means = set_size * posterior_cov_diag * posterior_means
    posterior_cov = np.diag(posterior_cov_diag)

    actual = teacher.calc_logp_likelihood(v, sets)

    for i, mean in enumerate(posterior_means):
        expected = gaussian(mean, posterior_cov).logpdf(v)

        assert actual[i] == expected


def test_calc_logp_prior(teacher, test_args):
    np.random.seed(1234)

    v = test_args['v']

    for set_size in [1, 2, 3]:
        sets = teacher.generate_teaching_sets(v, set_size)
        for batch in sets:
            logps = teacher.calc_logp_prior(v, batch, 'uniform')

            assert np.asarray(logps).sum() == 0

    with pytest.raises(Exception):
        sets = list(teacher.generate_teaching_sets(v, 1))
        teacher.calc_logp_prior(v, sets, 'something else')

    with pytest.raises(Exception):
        sets = list(teacher.generate_teaching_sets(v, 2))
        teacher.calc_logp_prior(v, sets, 'something else')


def test_calc_logp_marginal_likelihood(teacher):
    np.random.seed(1234)

    n_sets = 100

    logps_prior = np.random.uniform(0, -100, n_sets)
    logps_likelihood = np.random.uniform(0, -100, n_sets)
    logps = logps_prior + logps_likelihood
    expected = logsumexp(logps)

    actual = teacher.calc_logp_marginal_likelihood(logps)
    assert actual == expected


def score_teaching_sets(teacher, test_args):
    """
    Implemented in test_category_mean_teacher_integration.py and
                   test_category_mean_teacher_inference.py.
    """
    pass
