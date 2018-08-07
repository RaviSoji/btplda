import numpy as np
from itertools import combinations
from itertools import islice
from scipy.special import comb, logsumexp
from scipy.stats import multivariate_normal as gaussian


def calc_n_batches(N, set_size, n_sets, batch_size):
    total_sets = int(comb(N, set_size))

    if n_sets is None:
        n_sets = total_sets

    if batch_size is None:
        batch_size = n_sets

    return int(np.ceil(n_sets / batch_size)), total_sets


def generate_batches(sets, batch_size, n_batches, total_sets):
    assert batch_size is not None
    assert n_batches is not None
    assert batch_size != total_sets

    i = 0
    while i < n_batches:
        batch = list(islice(sets, batch_size))

        if len(batch) == 0:
            break

        yield batch
        i += 1


class MeanTeacher:
    def __init__(self, U_model, target_model):
        assert len(U_model.shape) == 2

        self.data = U_model
        self.N = self.data.shape[-2]
        self.prior_cov_diag = target_model.prior_params['cov_diag']

        assert U_model.shape[-1] == self.prior_cov_diag.shape[-1]

    def generate_teaching_sets(self, v, set_size,
                               n_sets=None, batch_size=None,
                               sequential_sampling=None):
        """ Returns a zip object. """
        assert set_size > 0
        assert sequential_sampling is None or \
               sequential_sampling == 'ascending' or \
               sequential_sampling == 'descending'

        idxs = np.arange(self.N)

        if sequential_sampling is not None:
            # Sequential_order == 'ascending'.
            logps = self.calc_logp_likelihood(v, idxs)
            idxs = np.argsort(logps)

            if sequential_sampling == 'descending':
                idxs = idxs[::-1]

        else:
            np.random.shuffle(idxs)

        sets = combinations(idxs, set_size)
        n_batches, \
        total_sets = calc_n_batches(self.N, set_size, n_sets, batch_size)

        if n_sets is not None:
            batch_size = min(batch_size, n_sets)

        if batch_size is None and n_sets is None:
            return list(sets)
    
        elif batch_size is None and n_sets is not None:
            return list(islice(sets, n_sets))

        elif batch_size == n_sets:
            return list(sets)

        elif total_sets < batch_size:
            return list(sets)

        else:
            return generate_batches(sets, batch_size, n_batches, n_sets)


    def calc_logp_likelihood(self, v, teaching_sets):
        """ Recall that the likelihood is the LEARNER's posterior. """
        if len(teaching_sets.shape) == 1:
            teaching_sets = teaching_sets[:, None]
            set_size = 1

        else:
            set_size = teaching_sets.shape[-1]

            assert len(teaching_sets.shape) > 1

        Psi_model = self.prior_cov_diag

        posterior_cov_diag = Psi_model / (set_size * Psi_model + 1)

        posterior_means = self.data[teaching_sets].sum(axis=-2)
        posterior_means = posterior_means * posterior_cov_diag

        # Since N(mean | posterior_mean, cov) = N(posterior_mean | mean, cov).
        logpdf = gaussian(v, np.diag(posterior_cov_diag)).logpdf

        return logpdf(posterior_means)

    def calc_logp_prior(self, v, teaching_sets, prior_type='uniform'):
        if prior_type is 'uniform':
            return 0  # Uniform Prior.

        else:
            raise NotImplementedError

    def calc_logp_marginal_likelihood(self, logps_likelihood_times_prior):
        return logsumexp(logps_likelihood_times_prior, axis=-1)

    def score_teaching_sets(self, v, teaching_sets, prior='uniform'):
        assert v.shape == self.data[0].shape
        assert type(norm_logps) == bool

        return self.calc_logp_likelihood(v, teaching_sets) + \
               self.calc_logp_prior(v, teaching_sets, prior)
