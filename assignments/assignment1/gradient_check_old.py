import numpy as np

CRED = '\033[91m'
CGREN = '\033[42m'
CEND = '\033[0m'
def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"
    assert analytic_grad.shape == x.shape
    #http://qaru.site/questions/15290759/implementing-a-naive-gradient-descent-check-using-numpy
    # We will go through every dimension of x and compute numeric
    # derivative for it

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    print('analytic grads= ',analytic_grad)
    print("X=", x)
    while not it.finished:
        ix = it.multi_index#x index
        print(CRED +'ix= ',ix, CEND)
        analytic_grad_at_ix = analytic_grad[ix]
        print('analgrad_at_ix= ',analytic_grad_at_ix)

        x_plus_h = x.copy()
        x_minus_h = x.copy()

        x_plus_h[ix] += delta
        x_minus_h[ix] -= delta
        print("X -", x_minus_h)
        print("X +", x_plus_h)
        numeric_grad_at_ix = (f(x_plus_h)[0] - f(x_minus_h)[0])/(2.*delta)

        # TODO compute value of numeric gradient of f to idx
        print(CRED + 'comparing numeric grad (calculated) ', numeric_grad_at_ix, " with analytical ", analytic_grad_at_ix, CEND)
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print(CRED + "-"*80 + CEND)
            print(CRED + "Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix) + CEND)
            print("*"*80)
            return False
        it.iternext()

    print(CGREN + "Gradient check passed!" + CEND)
    print("*"*60)
    return True




