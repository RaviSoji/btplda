import numpy as np

from itertools import combinations
from scipy.special import logsumexp
from scipy.stats import multivariate_normal as gaussian


def extract_logpdfs_array(posterior_predictive_params):
    means = []
    cov_diags = []
    labels = []

    for k, k_dict in posterior_predictive_params.items():
        means.append(k_dict['mean'])
        cov_diags.append(k_dict['cov_diag'])
        labels.append(k)

    all_logpdfs = []

    for mean, cov_diag in zip(means, cov_diags):  # For each category.
        category_logpdfs = []

        for mu, var in zip(mean, cov_diag):  # For each dimension.
            dimension_logpdf = gaussian(mu, var).logpdf
            category_logpdfs.append(dimension_logpdf)

        all_logpdfs.append(category_logpdfs)

    return all_logpdfs, labels


def extract_logpps_array(u_model, model, teaching_sets):
    pp_params = model.posterior_predictive_params
    logpdfs_array, labels = extract_logpdfs_array(pp_params)

    all_logpps = []

    for logpdfs_row in logpdfs_array:
        assert len(logpdfs_row) == u_model.shape[0]

        logpps_row = []

        for logpdf, u_dim in zip(logpdfs_row, u_model):
            logpps_row.append(logpdf(u_dim))

        all_logpps.append(logpps_row)

    all_logpps = np.asarray(all_logpps)[:, teaching_sets]

    return np.sum(all_logpps, axis=-1), labels


def normalize_logpps_array(logpps_array):
    """ Normalize probabilities for each dimension across categories. """
    assert len(logpps_array.shape) == 2

    norms = logsumexp(logpps_array, axis=-2)

    return logpps_array - norms


def label_to_idx(label, all_labels):
    return all_labels.index(label)


def to_image_space(u_model_vector, teaching_set, model):
    A = model.A
    m = model.m
    relevant_U_dims = model.relevant_U_dims

    u_vector = np.zeros(m.shape)
    u_vector[relevant_U_dims] = u_model_vector

    x_vector = np.matmul(A, u_vector)
    x_vector[teaching_set] + m[teaching_set]

    d_vector = model.transform(x_vector, from_space='X', to_space='D')
    d_vector = np.squeeze(d_vector)

    return d_vector


class PredictionTeacher:
    def __init__(self, u_model, target_model):
        assert len(u_model.shape) == 1

        self.datum = u_model
        self.model = target_model

    def gen_teaching_sets(self, set_size):
        n_dims = self.datum.shape[0]
        idxs = np.arange(n_dims)

        assert 0 < set_size <= n_dims

        sets = combinations(idxs, set_size)

        return np.asarray([teaching_set for teaching_set in sets])

    def calc_logp_prior(self, k, teaching_sets, prior='uniform'):
        if prior == 'uniform':
            return 0

        else:
            raise NotImplementedError

    def calc_logp_likelihood(self, k, teaching_sets):
        args = (self.datum, self.model, teaching_sets)
        logpps_array, categories = extract_logpps_array(*args)
        logpps_array = normalize_logpps_array(logpps_array)

        idx = label_to_idx(k, categories)

        return logpps_array[idx]

    def calc_logp_marginal_likelihood(self, logps_likelihood_times_prior):
        return logsumexp(logps_likelihood_times_prior, axis=-1)

    def build_filter(self, teaching_set):
        u_model_vector = np.zeros(self.datum.shape)
        u_model_vector[teaching_set] = self.datum[teaching_set]

        return to_image_space(u_model_vector, teaching_set, self.model)

    def apply_filter(self, datum, datum_filter):
        """ Convolves the image/datum and normalizes it. """
        assert datum.shape == datum_filter.shape

        highlighted_datum = datum * datum_filter

        return highlighted_datum / np.max(highlighted_datum)
