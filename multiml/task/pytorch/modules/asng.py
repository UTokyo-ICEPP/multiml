"""ASNG-NAS."""
import numpy as np
from .asng_util import ranking_based_utility_transformation


class AdaptiveSNG:
    """Adaptive Stochastic Natural Gradient for Categorical Distribution."""
    def __init__(self,
                 categories=None,
                 integers=None,
                 alpha=1.5,
                 delta_init=1.,
                 lam=2,
                 delta_max=np.inf,
                 init_theta_cat=None,
                 init_theta_int=None,
                 threshold=0.10,
                 patience=-1,
                 range_restriction=True):
        """AdaptiveSNG."""
        if categories is None and integers is None:
            return TypeError("one of categories or integers should be set")
        # Adaptive SG
        self.alpha = alpha  # threshold for adaptation
        self.delta_init = delta_init
        self.lam = lam  # lambda_theta
        self.delta_max = delta_max  # maximum Delta (can be np.inf)

        self.Delta = 1.
        self.gamma = 0.0  # correction factor
        self.delta = self.delta_init / self.Delta
        self.eps = self.delta

        # this is not in the original paper
        from .asng_util import ASNG_terminate_condition
        self.terminate = ASNG_terminate_condition(threshold=threshold, patience=patience)
        self.n_theta = 0

        # Categorical distribution
        if categories is not None:
            from .asng_util import asng_category
            self.cat = asng_category(categories, init_theta_cat, range_restriction)
            self.terminate.theta_cat_init(self.cat.theta.copy())
            self.n_theta += self.cat.get_n()
        else:
            self.cat = None

        # Normal distribution
        if integers is not None:
            from .asng_util import asng_integer
            self.int = asng_integer(integers, init_theta_int)
            self.terminate.theta_int_init(self.int.theta.copy())
            self.n_theta += self.cat.get_n()
        else:
            self.int = None

        self.s = np.zeros(self.n_theta)  # averaged stochastic natural gradient

    def get_lambda(self):
        return self.lam

    def check_converge(self):
        return self.terminate(self.cat.get_theta(), self.int.get_theta())

    def converge_counter(self):
        return self.terminate.counter

    def update_parameters(self, fnorm_cat, fnorm_int, hstack):
        self.delta = self.delta_init / self.Delta
        self.fnorm = np.sqrt(fnorm_cat + fnorm_int)
        self.eps = self.delta / (self.fnorm + 1e-9)
        self.beta = self.delta / (self.n_theta**0.5)
        self.s = (1 - self.beta) * self.s + np.sqrt(self.beta *
                                                    (2 - self.beta)) * hstack / self.fnorm
        self.gamma = (1 - self.beta)**2 * self.gamma + self.beta * (2 - self.beta)
        self.Delta *= np.exp(self.beta * (self.gamma - np.sum(self.s**2) / self.alpha))
        self.Delta = min(self.Delta, self.delta_max)

    def most_likely_value(self):
        return self.cat.get_most_likely(), self.int.get_most_likely()

    def get_thetas(self):
        return self.cat.get_theta(), self.int.get_theta()

    def set_thetas(self, theta_cat, theta_int):
        self.cat.set_theta(theta_cat)
        self.int.set_theta(theta_int)

    def sampling(self):
        c_cats = self.cat.sampling(self.lam)
        c_ints = self.int.sampling(self.lam)
        return c_cats, c_ints

    def update_theta(self, c_cat, c_int, losses):
        aru, idx = ranking_based_utility_transformation(losses, lam=self.lam)
        if np.all(aru == 0):
            return

        ## calculation natural gradient
        fnorm_cat, sl = self.cat.calc_theta(c_cat, aru, idx)
        fnorm_int, fdpara = self.int.calc_theta(c_int, aru, idx)
        hstack = np.hstack(tuple(sl, np.ravel(fdpara)))
        self.update_parameters(fnorm_cat, fnorm_int, hstack)

        # update theta
        self.cat.update_theta(self.eps)
        self.int.update_theta(self.eps)


class AdaptiveSNG_cat(AdaptiveSNG):
    def check_converge(self):
        return self.terminate(self.cat.get_theta(), None)

    def most_likely_value(self):
        return self.cat.get_most_likely(), None

    def get_thetas(self):
        return self.cat.get_theta(), None

    def set_thetas(self, theta_cat, theta_int):
        self.cat.set_theta(theta_cat)

    def sampling(self):
        c_cats = self.cat.sampling(self.lam)
        c_ints = [None] * self.lam
        return c_cats, c_ints

    def update_theta(self, c_cat, c_int, losses):
        aru, idx = ranking_based_utility_transformation(losses, lam=self.lam)
        if np.all(aru == 0):
            return

        ## calculation natural gradient
        fnorm_cat, sl = self.cat.calc_theta(c_cat, aru, idx)
        hstack = np.hstack(tuple(sl, ))
        self.update_parameters(fnorm_cat, 0.0, hstack)

        # update theta
        self.cat.update_theta(self.eps)


class AdaptiveSNG_int(AdaptiveSNG):
    def check_converge(self):
        return self.terminate(None, self.int.get_theta())

    def most_likely_value(self):
        return None, self.int.get_most_likely()

    def get_thetas(self):
        return None, self.int.get_theta()

    def set_thetas(self, theta_cat, theta_int):
        self.int.set_theta(theta_int)

    def sampling(self):
        c_cats = [None] * self.lam
        c_ints = self.int.sampling(self.lam)
        return c_cats, c_ints

    def update_theta(self, c_cat, c_int, losses):
        aru, idx = ranking_based_utility_transformation(losses, lam=self.lam)
        if np.all(aru == 0):
            return

        ## calculation natural gradient
        fnorm_int, fdpara = self.int.calc_theta(c_int, aru, idx)
        hstack = np.hstack(tuple(np.ravel(fdpara), ))
        self.update_parameters(0.0, fnorm_int, hstack)

        # update theta
        self.int.update_theta(self.eps)
