import numpy as np
from scipy import linalg


def foxgood(n):
    """
    FOXGOOD Test problem: severely ill-posed problem.

    [A,b,x] = foxgood(n)

    This is adapted from MATLAB codes by:
    Per Christian Hansen (2020). regtools (https://www.mathworks.com/matlabcentral/fileexchange/52-regtools).
    """
    h = 1/n
    t = h*(np.arange(n)[:, np.newaxis] + .5)
    a = h*np.sqrt(np.matmul(np.power(t, 2), np.ones((1, n))) + np.matmul(np.ones((n, 1)), np.power(t, 2).T))
    x = t
    b = (np.power(1+np.power(t, 2), 1.5) - np.power(t, 3)) / 3
    return a, b, x


def baart(n):
    """
    BAART Test problem: Fredholm integral equation of the first kind.

    [A,b,x] = baart(n)

    This is adapted from MATLAB codes by:
    Per Christian Hansen (2020). regtools (https://www.mathworks.com/matlabcentral/fileexchange/52-regtools).
    """
    if n % 2 != 0:
        raise Exception('The order n must be even.')

    hs = np.pi / (2*n)
    ht = np.pi / n
    c = 1/(3*np.sqrt(2))
    a = np.zeros((n, n))
    ihs = np.arange(n+1)[:, np.newaxis] * hs
    nh = n/2
    f3 = np.exp(ihs[1:n+1]) - np.exp(ihs[0:n])
    for j in range(1, n+1):
        f1 = f3
        co2 = np.cos((j-.5)*ht)
        co3 = np.cos(j*ht)
        f2 = (np.exp(ihs[1:n+1]*co2) - np.exp(ihs[0:n]*co2)) / co2
        if j == nh:
            f3 = hs*np.ones((n, 1))
        else:
            f3 = (np.exp(ihs[1:n+1]*co3) - np.exp(ihs[0:n]*co3)) / co3
        a[:, j-1:j] = c*(f1 + 4*f2 + f3)

    si = np.arange(.5, n+.5, .5)[:, np.newaxis]*hs
    si = np.divide(np.sinh(si), si)
    b = np.zeros((n, 1))
    b[0] = 1 + 4*si[0] + si[1]
    b[1:n] = si[1:2*n-2:2] + 4*si[2:2*n-1:2] + si[3:2*n:2]
    b = b*np.sqrt(hs)/3

    x = -np.diff(np.cos(np.arange(n+1)[:, np.newaxis]*ht), axis=0) / np.sqrt(ht)

    return a, b, x


def phillips(n):
    """
    PHILLIPS Test problem: Phillips' "famous" problem.

    [A,b,x] = phillips(n)

   This is adapted from MATLAB codes by:
    Per Christian Hansen (2020). regtools (https://www.mathworks.com/matlabcentral/fileexchange/52-regtools).
    """
    if n % 4 != 0:
        raise Exception('The order n must be a multiple of 4.')

    h = 12/n
    n4 = n//4
    r1 = np.zeros(n)
    c = np.cos(np.arange(-1, n4+1)*4*np.pi/n)
    r1[0:n4] = h + 9/(h*np.pi**2)*(2*c[1:n4+1] - c[0:n4] - c[2:n4+2])
    r1[n4:n4+1] = h/2 + 9/(h*np.pi**2)*(np.cos(4*np.pi/n)-1)
    a = linalg.toeplitz(r1)

    b = np.zeros((n, 1))
    c = np.pi/3
    for i in range(n//2+1, n+1):
        t1 = -6 + i*h
        t2 = t1 - h
        b[i-1] = t1*(6-np.abs(t1)/2) \
            + ((3-np.abs(t1)/2)*np.sin(c*t1) - 2/c*(np.cos(c*t1)-1))/c \
            - t2*(6-np.abs(t2)/2) \
            - ((3-np.abs(t2)/2)*np.sin(c*t2) - 2/c*(np.cos(c*t2)-1))/c
        b[n-i] = b[i-1]
    b = b/np.sqrt(h)

    x = np.zeros((n, 1))
    x[2*n4:3*n4] = (h + np.diff(np.sin(np.arange(0, 3+h/2, h)*c)[:, np.newaxis], axis=0)/c)/np.sqrt(h)
    x[n4:2*n4] = x[3*n4-1:2*n4-1:-1]

    return a, b, x


def get_l(n, d):
    """
    GET_L Compute discrete derivative operators.

    [L,W] = get_l(n,d)

    :param  n: int
    :param  d: int
    :return l: (n-d) * n numpy array
    """
    if d < 0:
        raise Exception('The order d must be nonnegative.')

    if d == 0:
        l = np.eye(n)
        return l

    # c = np.diff([1, -1], n=d-1, prepend=np.zeros(d-1), append=np.zeros(d-1))
    c = np.diff(np.hstack((np.zeros(d-1), [1, -1], np.zeros(d-1))), n=d-1)
    l = np.zeros((n-d, n))
    for i in range(d+1):
        np.fill_diagonal(l[:, i:], c[i])
    return l

