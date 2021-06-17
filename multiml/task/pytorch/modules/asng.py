""" ASNG-NAS
"""
import numpy as np


class AdaptiveSNG:
    """
    Adaptive Stochastic Natural Gradient for Categorical Distribution
    """
    def __init__(self,
                 categories=None,
                 integers=None,
                 alpha=1.5,
                 delta_init=1.,
                 lam=2,
                 delta_max=np.inf,
                 init_theta=None):
        """ AdaptiveSNG
        """
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

        # Categorical distribution
        if categories is not None:
            self.cat_n = np.sum(categories - 1)
            self.cat_d = len(categories)
            self.cat = categories
            self.cat_max = np.max(categories)
            self.theta_cat = np.zeros((self.cat_d, self.cat_max))
            # initialize theta by 1/C for each dimensions
            for i in range(self.cat_d):
                self.theta_cat[i, :self.cat[i]] = 1. / self.cat[i]
            # pad zeros to unused elements
            for i in range(self.cat_d):
                self.theta_cat[i, self.cat[i]:] = 0.
            # valid dimension size
            self.cat_d_valid = len(self.cat[self.cat > 1])
            if init_theta is not None:
                self.theta_cat = init_theta[0]
        else:
            self.cat_n = 0
            self.theta_cat = None

        # Normal distribution
        if integers is not None:
            self.int_n = 2 * len(integers)
            self.int_d = len(integers)
            self.int_min = np.array(integers)[:, 0]
            self.int_max = np.array(integers)[:, 1]
            self.int_std_max = (self.int_max - self.int_min) / 2.
            self.int_std_min = 1. / 4.
            # initialize theta
            self.theta_int = np.zeros((2, self.int_d))
            self.theta_int[0] = (self.int_max + self.int_min) / 2.
            self.theta_int[1] = (
                (self.int_max + self.int_min) / 2.)**2 + self.int_std_max**2

            if init_theta is not None:
                self.theta_int = init_theta[1]
        else:
            self.int_n = 0
            self.theta_int = None

        self.n_theta = self.cat_n + self.int_n
        self.s = np.zeros(self.n_theta)  # averaged stochastic natural gradient

    def get_lambda(self):
        return self.lam

    def most_likely_value(self):
        return self.most_likely_cat(), self.most_likely_int()

    def get_thetas(self):
        theta_cat = self.theta_cat
        theta_int = self.theta_int
        return theta_cat, theta_int

    def most_likely_cat(self):
        """ Get most likely categorical variables (one-hot)
        """
        if self.theta_cat is not None:
            c_cat = self.theta_cat.argmax(axis=1)
            T = np.zeros((self.cat_d, self.cat_max))
            for i, c in enumerate(c_cat):
                T[i, c] = 1
            return T
        return None

    def most_likely_int(self):
        if self.theta_int is not None:
            return self.theta_int[0]
        return None

    def sampling(self):
        c_cats, c_ints = [None] * self.lam, [None] * self.lam

        if self.theta_cat is not None:
            c_cats = self.sampling_cat()
        if self.theta_int is not None:
            c_ints = self.sampling_int()
        return c_cats, c_ints

    def sampling_cat(self):
        # Draw a sample from the categorical distribution (one-hot)
        rand = np.random.rand(self.lam, self.cat_d,
                              1)  # range of random number is [0, 1)
        cum_theta = self.theta_cat.cumsum(axis=1)  # (d, Cmax)
        # x[i, j] becomes 1 if cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        c_cat = (cum_theta - self.theta_cat <= rand) & (rand < cum_theta)
        return c_cat

    def sampling_int(self):

        c_int = np.empty((self.lam + 1, self.int_d))
        avg = self.theta_int[0]
        std = np.sqrt(self.theta_int[1] - avg**2)
        for i in range(int(np.round(lam / 2))):
            # Draw a sample from the normal distribution
            # symmetric sampling
            Z = np.random.randn(self.int_d)
            c_int[2 * i] = avg + std * Z
            c_int[2 * i + 1] = avg - std * Z
        return c_int[:lam]

    def update_theta(self, c_cat, c_int, losses, range_restriction=True):
        self.delta = self.delta_init / self.Delta

        # print(f'loss is {[l.item() for l in losses]}')
        aru, idx = self.utility(losses)

        if np.all(aru == 0):
            # If all the points have the same f-value,
            # nothing happens for theta and breaks.
            # In this case, we skip the rest of the code.
            # print(f'skip........')
            return

        fnorm_cat = 0.0
        fnorm_ord = 0.0
        hstack = []
        # print(self.theta_cat)
        if self.theta_cat is not None:
            # NG for categorical distribution
            ng_cat = np.mean(aru[:, np.newaxis, np.newaxis] *
                             (c_cat[idx] - self.theta_cat),
                             axis=0)
            # print(f'ng_cat is ')
            # print(f'{ng_cat}')
            # sqrt(F) * NG for categorical distribution
            sl = []
            for i, K in enumerate(self.cat):
                theta_i = self.theta_cat[i, :K - 1]
                theta_K = self.theta_cat[i, K - 1]
                s_i = 1. / np.sqrt(theta_i) * ng_cat[i, :K - 1]
                s_i += np.sqrt(theta_i) * ng_cat[i, :K - 1].sum() / (
                    theta_K + np.sqrt(theta_K))
                sl += list(s_i)
            sl = np.array(sl)
            fnorm_cat = np.sum(sl**2)
            hstack.append(sl)

        if self.theta_int is not None:
            # NG for normal distribution
            ng_int1 = np.mean(aru[:, np.newaxis] *
                              (c_int[idx] - self.theta_int[0]),
                              axis=0)
            ng_int2 = np.mean(aru[:, np.newaxis] *
                              (c_int[idx]**2 - self.theta_int[1]),
                              axis=0)
            dpara = np.vstack((ng_int1, ng_int2))

            # sqrt(F) * NG for normal distribution
            avg = self.theta_int[0]
            std = np.sqrt(self.theta_int[1] - avg**2)
            eigval = np.zeros((2, self.int_d))
            eigvec1 = np.zeros((2, self.int_d))
            eigvec2 = np.zeros((2, self.int_d))
            # Inverse Fisher (Ordinal)
            fb = 2 * avg
            fc = 4 * avg**2 + 2 * std**2
            eigval[0, :] = ((1. + fc) + np.sqrt((1 - fc)**2 + 4 * fb**2)) / 2.
            eigval[1, :] = ((1. + fc) - np.sqrt((1 - fc)**2 + 4 * fb**2)) / 2.
            mask = np.abs(fb) < 1e-8
            neg_mask = np.logical_not(mask)
            eigvec1[1, mask] = eigvec2[0, mask] = 0.
            eigvec1[0, mask] = eigvec2[1, mask] = 1.
            eigvec1[0, neg_mask] = eigvec2[0, neg_mask] = 1.
            eigvec1[1, neg_mask] = (eigval[0, neg_mask] - 1.) / fb[neg_mask]
            eigvec2[1, neg_mask] = (eigval[1, neg_mask] - 1.) / fb[neg_mask]
            eigvec1 /= np.linalg.norm(eigvec1, axis=0)
            eigvec2 /= np.linalg.norm(eigvec2, axis=0)
            eigval[0, :] *= std**2
            eigval[1, :] *= std**2

            # sqrt(F) * NG
            fdpara = np.zeros((2, self.int_d))  # sqrt(F) * dtheta
            fdpara[0, :] = eigvec1[0, :] * dpara[0] + eigvec1[1, :] * dpara[1]
            fdpara[1, :] = eigvec2[0, :] * dpara[0] + eigvec2[1, :] * dpara[1]
            fdpara[0, :] /= np.sqrt(eigval[0, :])
            fdpara[1, :] /= np.sqrt(eigval[1, :])
            fdpara = eigvec1 * fdpara[0] + eigvec2 * fdpara[1]
            fnorm_ord = np.sum(fdpara**2)
            hstack.append(np.ravel(fdpara))

        fnorm = np.sqrt(fnorm_cat + fnorm_ord)
        self.eps = self.delta / (fnorm + 1e-9)
        # print(self.eps * ng_cat)
        # print(self.theta_cat)
        # update
        if self.theta_cat is not None:
            self.theta_cat += self.eps * ng_cat
        else:
            self.theta_cat = None

        if self.theta_int is not None:
            self.theta_int += self.eps * dpara
        else:
            self.theta_int = None

        beta = self.delta / (self.n_theta**0.5)
        self.s = (1 - beta) * self.s + np.sqrt(beta * (2 - beta)) * np.hstack(
            tuple(hstack)) / fnorm
        self.gamma = (1 - beta)**2 * self.gamma + beta * (2 - beta)
        self.Delta *= np.exp(beta *
                             (self.gamma - np.sum(self.s**2) / self.alpha))
        self.Delta = min(self.Delta, self.delta_max)
        # print(f'eps = delta / fnorm  is {self.eps} = {self.delta} / {fnorm} ({self.Delta}, {beta})')

        if self.theta_cat is not None:
            # range restriction
            for i in range(self.cat_d):
                ci = self.cat[i]
                # Constraint for theta (minimum value of theta and sum of theta = 1.0)
                theta_min = 1. / (
                    self.cat_d_valid *
                    (ci - 1)) if range_restriction and ci > 1 else 1e-04
                self.theta_cat[i, :ci] = np.maximum(self.theta_cat[i, :ci],
                                                    theta_min)
                theta_sum = self.theta_cat[i, :ci].sum()
                tmp = theta_sum - theta_min * ci
                self.theta_cat[i, :ci] -= (theta_sum - 1.) * (
                    self.theta_cat[i, :ci] - theta_min) / tmp
                # Ensure the summation to 1
                self.theta_cat[i, :ci] /= self.theta_cat[i, :ci].sum()

        if self.theta_int is not None:
            self.theta_int[0] = np.clip(self.theta_int[0], self.int_min,
                                        self.int_max)
            self.theta_int[1] = np.clip(
                self.theta_int[1], self.theta_int[0]**2 + self.int_std_min**2,
                self.theta_int[0]**2 + self.int_std_max**2)
        # print(self.theta_cat)

    def utility(self, losses, rho=0.25, negative=True):
        """
        Ranking Based Utility Transformation

        w(f(x)) / lambda =
            1/mu  if rank(x) <= mu
            0     if mu < rank(x) < lambda - mu
            -1/mu if lambda - mu <= rank(x)

        where rank(x) is the number of at least equally good
        points, including it self.

        The number of good and bad points, mu, is ceil(lambda/4).
        That is,
            mu = 1 if lambda = 2
            mu = 1 if lambda = 4
            mu = 2 if lambda = 6, etc.

        If there exist tie points, the utility values are
        equally distributed for these points.
        """
        eps = 1e-14
        idx = np.argsort(losses)
        mu = int(np.ceil(self.lam * rho))
        _w = np.zeros(self.lam)
        _w[:mu] = 1 / mu
        _w[self.lam - mu:] = -1 / mu if negative else 0
        w = np.zeros(self.lam)
        istart = 0
        for i in range(losses.shape[0] - 1):
            if losses[idx[i + 1]] - losses[idx[i]] < eps * losses[idx[i]]:
                pass
            elif istart < i:
                w[istart:i + 1] = np.mean(_w[istart:i + 1])
                istart = i + 1
            else:
                w[i] = _w[i]
                istart = i + 1
        w[istart:] = np.mean(_w[istart:])
        return w, idx
