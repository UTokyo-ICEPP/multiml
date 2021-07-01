import numpy as np
""" ASNG category part """


class asng_category:
    def __init__(self,
                 categories=None,
                 init_theta=None,
                 range_restriction=True):

        self.range_restriction = range_restriction
        self.n = np.sum(categories - 1)
        self.d = len(categories)
        self.cat = categories
        self.max = np.max(categories)
        self.theta = np.zeros((self.d, self.max))
        # initialize theta by 1/C for each dimensions
        for i in range(self.d):
            self.theta[i, :self.cat[i]] = 1. / self.cat[i]
        # pad zeros to unused elements
        for i in range(self.d):
            self.theta[i, self.cat[i]:] = 0.
        # valid dimension size
        self.d_valid = len(self.cat[self.cat > 1])
        if init_theta is not None:
            self.theta = init_theta[0]

    def get_n(self):
        return self.n

    def get_theta(self):
        return self.theta.copy()

    def set_theta(self, theta):
        self.theta = theta

    def get_most_likely(self):
        """ Get most likely categorical variables (one-hot)
        """
        c_cat = self.theta.argmax(axis=1)
        T = np.zeros((self.d, self.max))
        for i, c in enumerate(c_cat):
            T[i, c] = 1
        return T

    def sampling(self, lam):
        # Draw a sample from the categorical distribution (one-hot)
        rand = np.random.rand(lam, self.d,
                              1)  # range of random number is [0, 1)
        cum_theta = self.theta.cumsum(axis=1)  # (d, Cmax)
        # x[i, j] becomes 1 if cum_theta[i, j] - theta[i, j] <= rand[i] < cum_theta[i, j]
        c_cat = (cum_theta - self.theta <= rand) & (rand < cum_theta)
        return c_cat

    def calc_theta(self, c_cat, aru, idx):
        self.ng = np.mean(aru[:, np.newaxis, np.newaxis] *
                          (c_cat[idx] - self.theta),
                          axis=0)
        # sqrt(F) * NG for categorical distribution
        sl = []
        for i, K in enumerate(self.cat):
            theta_i = self.theta[i, :K - 1]
            theta_K = self.theta[i, K - 1]
            s_i = 1. / np.sqrt(theta_i) * self.ng[i, :K - 1]
            s_i += np.sqrt(theta_i) * self.ng[i, :K - 1].sum() / (
                theta_K + np.sqrt(theta_K))
            sl += list(s_i)
        sl = np.array(sl)
        fnorm = np.sum(sl**2)
        return fnorm, sl

    def update_theta(self, eps):
        self.theta += eps * self.ng
        # range restriction
        for i in range(self.d):
            ci = self.cat[i]
            # Constraint for theta (minimum value of theta and sum of theta = 1.0)
            theta_min = 1. / (
                self.d_valid *
                (ci - 1)) if self.range_restriction and ci > 1 else 0.0
            self.theta[i, :ci] = np.maximum(self.theta[i, :ci], theta_min)
            theta_sum = self.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.theta[i, :ci] -= (theta_sum - 1.) * (self.theta[i, :ci] -
                                                      theta_min) / tmp
            # Ensure the summation to 1
            self.theta[i, :ci] /= self.theta[i, :ci].sum()


""" ASNG integer part """


class asng_integer:
    def __init__(self, integers=None, init_theta=None):
        self.n = 2 * len(integers)
        self.d = len(integers)
        self.min = np.array(integers)[:, 0]
        self.max = np.array(integers)[:, 1]
        self.std_max = (self.max - self.min) / 2.
        self.std_min = 1. / 4.
        # initialize theta
        self.theta = np.zeros((2, self.d))
        self.theta[0] = (self.max + self.min) / 2.
        self.theta[1] = ((self.max + self.min) / 2.)**2 + self.std_max**2

        if init_theta is not None:
            self.theta = init_theta[1]

    def get_n(self):
        return self.n

    def get_theta(self):
        return self.theta.copy()

    def set_theta(self, theta):
        self.theta = theta

    def get_most_likely(self):
        return self.theta[0]

    def sampling(self, lam):
        c_int = np.empty((lam + 1, self.d))
        avg = self.theta[0]
        std = np.sqrt(self.theta[1] - avg**2)
        for i in range(int(np.round(lam / 2))):
            # Draw a sample from the normal distribution
            # symmetric sampling
            Z = np.random.randn(self.d)
            c_int[2 * i] = avg + std * Z
            c_int[2 * i + 1] = avg - std * Z
        return c_int[:lam]

    def calc_theta(self, c_int, aru, idx):
        # NG for normal distribution
        ng_int1 = np.mean(aru[:, np.newaxis] * (c_int[idx] - self.theta[0]),
                          axis=0)
        ng_int2 = np.mean(aru[:, np.newaxis] * (c_int[idx]**2 - self.theta[1]),
                          axis=0)
        self.dpara = np.vstack((ng_int1, ng_int2))

        # sqrt(F) * NG for normal distribution
        avg = self.theta[0]
        std = np.sqrt(self.theta[1] - avg**2)
        eigval = np.zeros((2, self.d))
        eigvec1 = np.zeros((2, self.d))
        eigvec2 = np.zeros((2, self.d))
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
        fdpara = np.zeros((2, self.d))  # sqrt(F) * dtheta
        fdpara[0, :] = eigvec1[0, :] * self.dpara[0] + eigvec1[
            1, :] * self.dpara[1]
        fdpara[1, :] = eigvec2[0, :] * self.dpara[0] + eigvec2[
            1, :] * self.dpara[1]
        fdpara[0, :] /= np.sqrt(eigval[0, :])
        fdpara[1, :] /= np.sqrt(eigval[1, :])
        fdpara = eigvec1 * fdpara[0] + eigvec2 * fdpara[1]
        fnorm_int = np.sum(fdpara**2)

        return fnorm_int, fdpara

    def update_theta(self):
        self.theta += eps * self.dpara
        self.theta[0] = np.clip(self.theta[0], self.min, self.max)
        self.theta[1] = np.clip(self.theta[1],
                                self.theta[0]**2 + self.std_min**2,
                                self.theta[0]**2 + self.std_max**2)


def ranking_based_utility_transformation(losses, lam, rho=0.25, negative=True):
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
    mu = int(np.ceil(lam * rho))
    _w = np.zeros(lam)
    _w[:mu] = 1 / mu
    _w[lam - mu:] = -1 / mu if negative else 0
    w = np.zeros(lam)
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


class ASNG_terminate_condition:
    def __init__(self, threshold, patience):
        self.counter = 0
        self.threshold = threshold
        self.patience = patience

    def theta_cat_init(self, theta):
        self.theta_cat_pre = theta

    def theta_int_init(self, theta):
        self.theta_int_pre = theta

    def __call__(self, theta_cat, theta_int):
        if self.patience < 0:
            return True

        is_converge = True

        if theta_cat is not None:
            delta_cat = self.theta_cat_pre - theta_cat
            is_converge *= np.all(np.fabs(delta_cat) < self.threshold)
            self.theta_cat_pre = theta_cat.copy()

        if theta_int is not None:
            delta_int = self.theta_int_pre - theta_int
            is_converge *= np.all(delta_int < self.threshold)
            self.theta_int_pre = theta_int.copy()

        if is_converge:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter > self.patience:
            return True
        return False
