import numpy as np


def _get_chatterjee_coeff(y_ranks):
    s = 0
    for i, ri in enumerate(y_ranks[1:]):
        s += np.abs(ri - y_ranks[i - 1])
    s *= 3
    s /= len(y_ranks) ** 2 - 1
    s = 1 - s
    return s


def get_chatterjee_coefficient_noTies(X, Y):
    x_sort_inds = np.argsort(X)
    x_sorted_x = X[x_sort_inds]
    y_sorted_x = Y[x_sort_inds]
    y_ranks = np.argsort(y_sorted_x)
    return _get_chatterjee_coeff(y_ranks)


def get_chatterjee_coefficient_yesTies(X, Y, num_trials=1000):
    coeffs = []
    for i in range(num_trials):
        # shuffle
        perm_inds = np.random.shuffle(np.arange(len(X)))
        X = X[perm_inds]
        Y = Y[perm_inds]

        # get coeff
        coeffs.append(get_chatterjee_coefficient_noTies(X, Y))
    return np.mean(coeffs)


def get_chatterjee_coefficient(X, Y):
    """
    https://arxiv.org/pdf/1909.10140.pdf
    """
    X = np.round(X, decimals=4)
    Y = np.round(Y, decimals=4)
    unq_xs = np.unique(X)
    if len(unq_xs) == len(X):
        return get_chatterjee_coefficient_noTies(X, Y)
    else:
        return get_chatterjee_coefficient_yesTies(X, Y)


def get_edit_distance_noTies(X, Y):
    pass


def get_edit_distance_yesTies(X, Y):
    pass


def get_edit_distance(X, Y):
    """
    Find number of edits required to go from one ranked list to the other
    """
    X = np.round(X, decimals=4)
    Y = np.round(Y, decimals=4)
    unq_xs = np.unique(X)
    if len(unq_xs) == len(X):
        return get_edit_distance_noTies(X, Y)
    else:
        return get_edit_distance_yesTies(X, Y)
