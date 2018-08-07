import numpy as np
import pytest

from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.special import comb, logsumexp
from scipy.stats import multivariate_normal as gaussian

from .category_mean_teacher_fixtures import teacher, test_args
from .category_mean_teacher_fixtures import classifier_and_latent_data
from .category_mean_teacher_fixtures import data_dictionary


def test_score_teaching_sets(teacher, test_args):
    np.random.seed(1234)

    v = test_args['v']

    def test_set_size(set_size, v):
        ordering = None

        sets = list(teacher.generate_teaching_sets(v, set_size, ordering))
        sets = np.asarray(sets)

        # Norm_logps set to False.
        logps = teacher.calc_logp_teaching_set(v, sets, norm_logps=False)
        actual = logps - teacher.calc_logp_prior(v, sets, 'uniform')
        actual -= teacher.calc_logp_likelihood(v, sets)
    
        expected = np.zeros(actual.shape)
        assert_allclose(actual, expected, rtol=1e-20)

        # Norm_logps set to True.
        logps = teacher.calc_logp_teaching_set(v, sets, norm_logps=True)
        assert_allclose(np.exp(logps).sum(), 1)

        logps_like = teacher.calc_logp_prior(v, sets, 'uniform')
        logps_prior = teacher.calc_logp_likelihood(v, sets)
        norm = teacher.calc_logp_marginal_likelihood(logps_like, logps_prior)

        actual = logps - logps_like - logps_prior + norm
    
        expected = np.zeros(actual.shape)
        assert_allclose(actual, expected, rtol=1e-20)

    test_set_size(1, v)
    test_set_size(2, v)

    # Test ordering. Note that you can only do this test under set_size = 1.
    ordering = 'ascending'

    sets = list(teacher.generate_teaching_sets(v, 1, ordering))
    sets = np.asarray(sets)
    logps_a = teacher.calc_logp_teaching_set(v, sets, norm_logps=False)

    ordering = 'descending'
    sets = list(teacher.generate_teaching_sets(v, 1, ordering))
    sets = np.asarray(sets)
    logps_b = teacher.calc_logp_teaching_set(v, sets, norm_logps=False)

    assert_array_equal(logps_a, logps_b[::-1])

    ordering = 'ascending'

    sets = list(teacher.generate_teaching_sets(v, 1, ordering))
    sets = np.asarray(sets)
    logps_a = teacher.calc_logp_teaching_set(v, sets, norm_logps=True)

    ordering = 'descending'
    sets = list(teacher.generate_teaching_sets(v, 1, ordering))
    sets = np.asarray(sets)
    logps_b = teacher.calc_logp_teaching_set(v, sets, norm_logps=True)

    assert_array_equal(logps_a, logps_b[::-1])

    # Test that random order still yields the same probabilities.
    ordering = None
    set_size = 2

    sets = list(teacher.generate_teaching_sets(v, set_size, ordering))
    sets = np.asarray(sets)
    logps_a = teacher.calc_logp_teaching_set(v, sets, norm_logps=False)

    sets = list(teacher.generate_teaching_sets(v, set_size, ordering))
    sets = np.asarray(sets)
    logps_b = teacher.calc_logp_teaching_set(v, sets, norm_logps=False)

    assert not np.array_equal(logps_a, logps_b)

    sorted_logps_a = np.sort(logps_a)
    sorted_logps_b = np.sort(logps_b)

    assert_array_equal(sorted_logps_a, sorted_logps_b)

    ordering = None

    sets = list(teacher.generate_teaching_sets(v, set_size, ordering))
    sets = np.asarray(sets)
    logps_a = teacher.calc_logp_teaching_set(v, sets, norm_logps=True)

    sets = list(teacher.generate_teaching_sets(v, set_size, ordering))
    sets = np.asarray(sets)
    logps_b = teacher.calc_logp_teaching_set(v, sets, norm_logps=True)

    assert not np.array_equal(logps_a, logps_b)

    sorted_logps_a = np.sort(logps_a)
    sorted_logps_b = np.sort(logps_b)

    assert_array_equal(sorted_logps_a, sorted_logps_b)
