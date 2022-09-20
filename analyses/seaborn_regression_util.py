import numbers
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression

def _handle_random_seed(seed=None):
    """Given a seed in one of many formats, return a random number generator.
    Generalizes across the numpy 1.17 changes, preferring newer functionality.
    """
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        try:
            # General interface for seeding on numpy >= 1.17
            rng = np.random.default_rng(seed)
        except AttributeError:
            # We are on numpy < 1.17, handle options ourselves
            if isinstance(seed, (numbers.Integral, np.integer)):
                rng = np.random.RandomState(seed)
            elif seed is None:
                rng = np.random.RandomState()
            else:
                err = "{} cannot be used to seed the randomn number generator"
                raise ValueError(err.format(seed))
    return rng

def bootstrap(*args, **kwargs):
    """Resample one or more arrays with replacement and store aggregate values.
    Positional arguments are a sequence of arrays to bootstrap along the first
    axis and pass to a summary function.
    Keyword arguments:
        n_boot : int, default 10000
            Number of iterations
        axis : int, default None
            Will pass axis to ``func`` as a keyword argument.
        units : array, default None
            Array of sampling unit IDs. When used the bootstrap resamples units
            and then observations within units instead of individual
            datapoints.
        func : string or callable, default np.mean
            Function to call on the args that are passed in. If string, tries
            to use as named method on numpy array.
        seed : Generator | SeedSequence | RandomState | int | None
            Seed for the random number generator; useful if you want
            reproducible resamples.
    Returns
    -------
    boot_dist: array
        array of bootstrapped statistic values
    """
    # Ensure list of arrays are same length
    if len(np.unique(list(map(len, args)))) > 1:
        raise ValueError("All input arrays must have the same length")
    n = len(args[0])

    # Default keyword arguments
    n_boot = kwargs.get("n_boot", 10000)
    func = kwargs.get("func", np.mean)
    axis = kwargs.get("axis", None)
    units = kwargs.get("units", None)
    random_seed = kwargs.get("random_seed", None)
    if random_seed is not None:
        msg = "`random_seed` has been renamed to `seed` and will be removed"
        warnings.warn(msg)
    seed = kwargs.get("seed", random_seed)
    if axis is None:
        func_kwargs = dict()
    else:
        func_kwargs = dict(axis=axis)

    # Initialize the resampler
    rng = _handle_random_seed(seed)

    # Coerce to arrays
    args = list(map(np.asarray, args))
    if units is not None:
        units = np.asarray(units)

    # Allow for a function that is the name of a method on an array
    if isinstance(func, str):
        def f(x):
            return getattr(x, func)()
    else:
        f = func

    # Handle numpy changes
    try:
        integers = rng.integers
    except AttributeError:
        integers = rng.randint

    # Do the bootstrap
    if units is not None:
        return _structured_bootstrap(args, n_boot, units, f,
                                     func_kwargs, integers)

    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(f(*sample, **func_kwargs))
    return np.array(boot_dist)


def ci(a, which=95, axis=None):
    """Return a percentile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.percentile(a, p, axis)

def fit_fast(x,y, grid):
    """Low-level regression and prediction using linear algebra."""
    def reg_func(_x, _y):
        return np.linalg.pinv(_x).dot(_y)

    X, y = np.c_[np.ones(len(x)), x], y
    grid = np.c_[np.ones(len(grid)), grid]
    yhat = grid.dot(reg_func(X, y))
    if ci is None:
        return yhat, None

    beta_boots = bootstrap(X, y,
                                func=reg_func,
                                n_boot=1000,
                                units=None,
                                seed=None).T
    yhat_boots = grid.dot(beta_boots).T
    return yhat, yhat_boots


def fit_fast_log(x, y, grid):

    def reg_func(_x, _y):
        x = np.log10(_x)
        x = np.log10(_y)
        clf = LinearRegression().fit(x.reshape(-1,1), y.reshape(-1,1))
        params = [clf.coef_, clf.intercept_]

        return np.power(np.ones(x.shape) * 10, params[0]*x+params[1])

    yhat = reg_func(x,y)
    if ci is None:
        return yhat, None

    yhat_boots = bootstrap(x, y,
                                func=reg_func,
                                n_boot=1000,
                                units=None,
                                seed=None).T
    return yhat, yhat_boots
