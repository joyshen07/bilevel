import numpy as np
import matplotlib.pyplot as plt
import gzip
import scipy.sparse
from itertools import groupby
import gurobipy as gp
import time
import os


def get_labels():
    b = -1*np.ones((50000, 1))
    with gzip.open('rcv1/rcv1-v2.topics.qrels.gz', 'r') as file:
        for line in file:
            category, did, _ = line.split()
            did = int(did.decode()) - 26151
            if 0 <= did < 50000 and category[0] == 77:
                b[did, 0] = 1
    np.save('data/data_b', b)


def get_features():
    tokens = set()
    with gzip.open('rcv1/lyrl2004_tokens_test_pt0.dat.gz', 'r') as file:
        for nonempty, sample in groupby(file, lambda line: line != b'\n' and line[0] != 46):
            if nonempty:
                for token in sample:
                    token = token.split()
                    tokens.update(token)
    tokens = {j: i for i, j in enumerate(tokens)}
    a = np.zeros((50000, len(tokens)))
    with gzip.open('rcv1/lyrl2004_tokens_test_pt0.dat.gz', 'r') as file:
        for line in file:
            if line[0:2] == b'.I':
                _, did = line.split()
                did = int(did.decode()) - 26151
            elif line[0:2] != b'.W':
                line = line.split()
                token = [tokens[x] for x in line]
                a[did, token] = 1  # binary
    a = scipy.sparse.csr_matrix(a, dtype='int8')
    scipy.sparse.save_npz('data/data_a', a)


def algo1(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
          stepsize_inner, stepsize_bi, stepsize_dual, slater,
          initial_inner, initial_bi, initial_dual, n_iter, pre_iter=0, start_ave=-1):

    z0 = initial_inner
    x0 = initial_bi
    y0 = initial_dual

    if start_ave < 0:
        start_ave = n_iter
    u0 = 0*z0
    w0 = 0*x0

    iters_inner_1 = [inner_obj(x0)]
    iters_outer_1 = [outer_obj(x0)]
    iters_inner_md = [inner_obj(z0)]
    iters_outer_md = [outer_obj(z0)]
    iters_time_1 = [0]
    iters_time_md = [0]

    for i in range(pre_iter):
        z1 = proj(z0 - stepsize_inner * inner_obj_grad(z0))
        z0 = z1

    sol_time_md = 0
    sol_time_1 = 0

    for i in range(n_iter):
        tic = time.perf_counter()

        z1 = proj( z0 - stepsize_inner * inner_obj_grad(z0) )
        toc_md = time.perf_counter()

        dx = inner_obj_grad(x0)
        x1 = proj(x0 - stepsize_bi * (outer_obj_grad(x0) + y0 * dx))
        y1 = max( 0, y0 + stepsize_dual *
                  ( inner_obj(x0) - inner_obj(z1) - slater + np.dot(dx, x1-x0) ) )

        z0 = z1
        x0 = x1
        y0 = y1

        toc_1 = time.perf_counter()
        sol_time_md += (toc_md - tic)
        sol_time_1  += (toc_1  - tic)

        if i > start_ave:
            u0 = (u0*(i-start_ave-1) + z1)/(i-start_ave)
            w0 = (w0*(i-start_ave-1) + x1)/(i-start_ave)
        else:
            u0 = z0
            w0 = x0

        if (i+1) % (n_iter // 100) == 0 or i < 100:
            print(f'{i+1:5}-th iteration, '
                  f'inner est. = {inner_obj(u0):10.6f}, '
                  f'inner obj. = {inner_obj(w0):10.6f}, '
                  f'outer est. = {outer_obj(u0):8.4f}, '
                  f'outer obj. = {outer_obj(w0):8.4f}, '
                  f'lambda = {y0:8.2f}, ')

        # record the iterations for plotting
        iters_inner_1.append(inner_obj(w0))
        iters_outer_1.append(outer_obj(w0))
        iters_inner_md.append(inner_obj(u0))
        iters_outer_md.append(outer_obj(u0))
        iters_time_1.append(sol_time_1)
        iters_time_md.append(sol_time_md)

    # print('Solution time: ', sol_time)

    return iters_inner_1, iters_outer_1, iters_inner_md, iters_outer_md, \
        iters_time_1, iters_time_md


def MirrorDescent(inner_obj, inner_obj_grad, outer_obj, proj,
                  stepsize_inner,
                  initial_inner, n_iter):

    z0 = initial_inner

    u0 = 0*z0

    iters_inner_md = [inner_obj(z0)]
    iters_outer_md = [outer_obj(z0)]

    # sol_time = 0

    for i in range(n_iter):
        # tic = time.perf_counter()

        z1 = proj( z0 - stepsize_inner * inner_obj_grad(z0) )
        z0 = z1

        u0 = (u0*i + z1)/(i+1)

        # toc = time.perf_counter()
        # sol_time += (toc - tic)

        if (i+1) % (n_iter // 100) == 0 or i < 100:
            print(f'{i+1:5}-th iteration, '
                  f'inner est. = {inner_obj(z0):10.6f}, '
                  f'outer est. = {outer_obj(z0):8.4f}, ')

        # record the iterations for plotting
        iters_inner_md.append(inner_obj(u0))
        iters_outer_md.append(outer_obj(u0))

    # print('Solution time: ', sol_time)

    return iters_inner_md, iters_outer_md


def IRIG(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
         gamma, lbd, r, eps,
         initial, n_iter):

    x0 = initial
    w0 = x0
    s0 = gamma ** r

    gamma1 = gamma
    lbd1 = lbd

    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]
    iters_time = [0]

    sol_time = 0

    for i in range(n_iter):
        tic = time.perf_counter()

        x1 = proj( x0 - gamma1 * ( inner_obj_grad(x0) + lbd1 * outer_obj_grad(x0) ) )

        gamma1 = gamma/(i+2)**(0.5+0.5*eps)
        lbd1 = lbd/(i+2)**(0.5-eps)

        s1 = s0 + gamma1 ** r
        w0 = (s0*w0 + (gamma1 ** r) * x1) / s1

        s0 = s1
        x0 = x1

        toc = time.perf_counter()
        sol_time += (toc - tic)

        if (i+1) % (n_iter // 100) == 0 or i < 100:
            print(f'{i+1:5}-th iteration, '
                  f'inner est. = {inner_obj(w0):10.6f}, '
                  f'outer est. = {outer_obj(w0):8.4f}, ')

        # record the iterations for plotting
        iters_inner.append(inner_obj(w0))
        iters_outer.append(outer_obj(w0))
        iters_time.append(sol_time)

    return iters_inner, iters_outer, iters_time


def BiGSAM(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
           t, s, alpha,
           initial, n_iter):

    x0 = initial

    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]
    iters_time = [0]

    sol_time = 0

    for i in range(n_iter):
        tic = time.perf_counter()

        y1 = proj(x0 - t * inner_obj_grad(x0))
        z1 = x0 - s * outer_obj_grad(x0)
        x1 = alpha/(i+1) * z1 + (1-alpha/(i+1)) * y1
        x0 = x1

        toc = time.perf_counter()
        sol_time += (toc - tic)

        if (i+1) % (n_iter // 100) == 0 or i < 100:
            print(f'{i+1:5}-th iteration, '
                  f'inner est. = {inner_obj(x0):10.6f}, '
                  f'outer est. = {outer_obj(x0):8.4f}, ')

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))
        iters_time.append(sol_time)

    return iters_inner, iters_outer, iters_time


def helou(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
          initial, n_iter):

    x0 = initial

    lmd = 3
    mu = 1
    y1 = x0 - lmd * inner_obj_grad(x0)
    z1 = x0 - mu * outer_obj_grad(y1)
    x1 = proj(z1)
    mu = np.linalg.norm(x0-y1) / np.linalg.norm(y1-z1) * 0.01

    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]
    iters_time = [0]

    sol_time = 0

    for i in range(n_iter):
        tic = time.perf_counter()

        y1 = x0 - lmd / np.power(i+1, 0.1) * inner_obj_grad(x0)
        z1 = y1 - mu / (i+1) * outer_obj_grad(y1)
        x0 = proj(z1)

        toc = time.perf_counter()
        sol_time += (toc - tic)

        if (i+1) % (n_iter // 100) == 0 or i < 100:
            print(f'{i+1:5}-th iteration, '
                  f'inner est. = {inner_obj(x0):10.6f}, '
                  f'outer est. = {outer_obj(x0):8.4f}, ')

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))
        iters_time.append(sol_time)

    return iters_inner, iters_outer, iters_time


def solodov(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
            alpha, theta, eta,
            initial, n_iter):

    x0 = initial

    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]
    iters_time = [0]

    sol_time = 0

    for i in range(n_iter):
        tic = time.perf_counter()

        # select sigma_i = 1/i
        phi = outer_obj(x0) / (i+1) + inner_obj(x0)
        phi_grad = outer_obj_grad(x0) / (i+1) + inner_obj_grad(x0)
        while True:
            x2 = proj(x0 - alpha * phi_grad)
            phi2 = outer_obj(x2) / (i+1) + inner_obj(x2)
            if phi2 <= phi + theta * np.dot(phi_grad, x2-x0) + 1e-6:
                break
            else:
                alpha *= eta
        x1 = x2
        x0 = x1

        toc = time.perf_counter()
        sol_time += (toc - tic)

        if (i+1) % (n_iter // 100) == 0 or i < 100:
            print(f'{i+1:5}-th iteration, '
                  f'inner est. = {inner_obj(x0):10.6f}, '
                  f'outer est. = {outer_obj(x0):8.4f}, '
                  f'alpha = {alpha}')

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))
        iters_time.append(sol_time)

    return iters_inner, iters_outer, iters_time


def text_cls_prob(full):

    # load data
    b = np.load('data/data_b.npy')
    a = scipy.sparse.load_npz('data/data_a.npz')
    a = scipy.sparse.hstack((a, np.ones((a.shape[0], 1))))
    m = b.size
    if not full:
        a = a.tocsr()[:1000, :1000]
        b = b[:1000]
        m = 1000
    b = b.flatten()

    def cls_inner_func(x):
        return np.sum(np.maximum(1 - b * (a @ x), 0)) / m

    def cls_inner_grad(x):
        y = 1 - b * (a @ x)
        y = y.flatten()
        dx = a.multiply(-b[:, np.newaxis]).tocsr()[y > 0, :]
        dx = np.sum(dx, 0).T.getA() / m
        return dx.flatten()

    def cls_outer_func(x):
        return np.linalg.norm(x, 1)

    def cls_outer_grad(x):
        return np.sign(x)

    def proj_unconstrained(x):
        return x

    def proj_l2ball(x):
        radius = 100
        norm = np.linalg.norm(x)
        if norm > radius:
            return x / norm * radius
        else:
            return x

    max_iter = 50000

    print('Running Algorithm 1...')
    # tic = time.perf_counter()
    iters_inner_1, iters_outer_1, iters_inner_md, iters_outer_md, iters_time_1, iters_time_md = \
        algo1(cls_inner_func, cls_inner_grad, cls_outer_func, cls_outer_grad, proj_l2ball,
                stepsize_inner=1, stepsize_bi=1e-4, stepsize_dual=1e3, slater=1e-4,
                initial_inner=np.zeros(a.shape[1]), initial_bi=np.zeros(a.shape[1]), initial_dual=0, n_iter=max_iter,
                start_ave=0)
    # toc = time.perf_counter()
    # print('Solution time (sec) = ' + f'{toc-tic}\n')

    # print('Running Mirror Descent...')
    # tic = time.perf_counter()
    # iters_inner_md, iters_outer_md = \
    #     MirrorDescent(cls_inner_func, cls_inner_grad, cls_outer_func, proj_l2ball,
    #                   stepsize_inner=1,
    #                   initial_inner=np.zeros(a.shape[1]), n_iter=max_iter)
    # toc = time.perf_counter()
    # print('Solution time (sec) = ' + f'{toc-tic}\n')

    print('Running IR-IG...')
    iters_inner_irig, iters_outer_irig, iters_time_irig = \
        IRIG(cls_inner_func, cls_inner_grad, cls_outer_func, cls_outer_grad, proj_l2ball,
             gamma=.1, lbd=.1, r=-10, eps=0,
             initial=np.zeros(a.shape[1]), n_iter=max_iter)

    print('Running BiG-SAM...')
    iters_inner_bigsam, iters_outer_bigsam, iters_time_bigsam = \
        BiGSAM(cls_inner_func, cls_inner_grad, cls_outer_func, cls_outer_grad, proj_l2ball,
               t=.01, s=.01, alpha=.5,
               initial=np.zeros(a.shape[1]), n_iter=max_iter)

    print('Running Helou\'s algorithm...')
    iters_inner_helou, iters_outer_helou, iters_time_helou = \
        helou(cls_inner_func, cls_inner_grad, cls_outer_func, cls_outer_grad, proj_l2ball,
              initial=np.zeros(a.shape[1]), n_iter=max_iter)

    # # save data
    # np.save('textcls_algo1_inner', iters_inner_1)
    # np.save('textcls_algo1_outer', iters_outer_1)
    # np.save('textcls_algo1_time', iters_time_1)
    # np.save('textcls_md_inner', iters_inner_md)
    # np.save('textcls_md_outer', iters_outer_md)
    # np.save('textcls_md_time', iters_time_md)
    # np.save('textcls_irig_inner', iters_inner_irig)
    # np.save('textcls_irig_outer', iters_outer_irig)
    # np.save('textcls_irig_time', iters_time_irig)
    # np.save('textcls_bigsam_inner', iters_inner_bigsam)
    # np.save('textcls_bigsam_outer', iters_outer_bigsam)
    # np.save('textcls_bigsam_time', iters_time_bigsam)
    # np.save('textcls_helou_inner', iters_inner_helou)
    # np.save('textcls_helou_outer', iters_outer_helou)
    # np.save('textcls_helou_time', iters_time_helou)
    # print('Raw data saved as npy files')

    # # solve by GUROBI
    # grb_inner, grb_outer = 0, 0
    # if not full:
    #     grb_inner, grb_outer = text_cls_gurobi()

    n_sample = 1000
    iters_inner_list = [iters_inner_1, iters_inner_md, iters_inner_irig, iters_inner_bigsam, iters_inner_helou]
    iters_outer_list = [iters_outer_1, iters_outer_md, iters_outer_irig, iters_outer_bigsam, iters_outer_helou]
    iters_time_list = [iters_time_1, iters_time_md, iters_time_irig, iters_time_bigsam, iters_time_helou]
    label_list = ['Algorithm 1', 'Mirror Descent', 'IR-IG', 'BiG-SAM', 'Helou']
    name_list = ['algo1', 'md', 'irig', 'bigsam', 'helou']
    plt.rc('text', usetex=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    for iters, label, name in zip(iters_inner_list, label_list, name_list):
        tab = []
        for i in range(n_sample):
            ind = np.floor(max_iter / n_sample * i).astype(int)
            tab.append([ind, iters[ind]])
        ax1.plot(np.asarray(tab)[:,0], np.asarray(tab)[:,1], label=label)
        np.savetxt('output/textcls_' + name + '_inner.csv', np.asarray(tab))
    ax1.set_xlabel(r'iteration $t$')
    ax1.set_ylabel(r'$\eta(\bar x_t)$')
    # plt.legend()

    for iters, label, name in zip(iters_outer_list, label_list, name_list):
        tab = []
        for i in range(n_sample):
            ind = np.floor(max_iter / n_sample * i).astype(int)
            tab.append([ind, iters[ind]])
        ax2.plot(np.asarray(tab)[:,0], np.asarray(tab)[:,1], label=label)
        np.savetxt('output/textcls_' + name + '_outer.csv', np.asarray(tab))
    ax2.set_xlabel(r'iteration $t$')
    ax2.set_ylabel(r'$\phi(\bar x_t)$')
    # plt.legend()
    # fig.suptitle(r'$\gamma_0=10^{-4}$, $\beta_0=10^3$, $r=T^{-1/3}$')

    for iters, label, name in zip(iters_time_list, label_list, name_list):
        tab = []
        for i in range(n_sample):
            ind = np.floor(max_iter / n_sample * i).astype(int)
            tab.append([ind, iters[ind]])
        ax3.plot(np.asarray(tab)[:,0], np.asarray(tab)[:,1], label=label)
        np.savetxt('output/textcls_' + name + '_time.csv', np.asarray(tab))
    ax3.set_xlabel(r'iteration $t$')
    ax3.set_ylabel(r'solution time')
    plt.legend()
    plt.show()


def text_cls_gurobi(full=False):
    # load data
    b = np.load('../data/data_b.npy')
    a = scipy.sparse.load_npz('../data/data_a.npz')
    a = scipy.sparse.hstack((a, np.ones((a.shape[0], 1))))
    if not full:
        a = a.tocsr()[:1000, :1000]
        b = b[:1000]
        n = 1000
        m = 1000
    else:
        n = 50000
        m = 70413

    # solve the inner problem by gurobipy
    prob_inner = gp.Model()
    x = prob_inner.addMVar(m, lb=-gp.GRB.INFINITY)
    y = prob_inner.addMVar(n, lb=-gp.GRB.INFINITY)
    z = prob_inner.addMVar(n, lb=0)
    prob_inner.setObjective(z.sum() / n, gp.GRB.MINIMIZE)
    prob_inner.addConstr(y == a @ x)
    prob_inner.addConstr(z >= 1 - np.diag(b[:, 0]) @ y)
    prob_inner.update()
    prob_inner.optimize()
    print(f'\ninner obj. value = ', prob_inner.objVal)

    # solve the outer problem by gurobipy
    prob_outer = gp.Model()
    x = prob_outer.addMVar(m, lb=-gp.GRB.INFINITY)
    w = prob_outer.addMVar(m, lb=-gp.GRB.INFINITY)
    y = prob_outer.addMVar(n, lb=-gp.GRB.INFINITY)
    z = prob_outer.addMVar(n, lb=0)
    prob_outer.setObjective(w.sum(), gp.GRB.MINIMIZE)
    prob_outer.addConstr(y == a @ x)
    prob_outer.addConstr(z >= 1 - np.diag(b[:, 0]) @ y)
    prob_outer.addConstr(z.sum() / n <= prob_inner.objVal * (1+0))
    prob_outer.addConstr(w >= -x)
    prob_outer.addConstr(w >= x)
    prob_outer.update()
    prob_outer.optimize()
    print(f'outer obj. value = ', prob_outer.objVal)

    return prob_inner.objVal, prob_outer.objVal


if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    text_cls_prob(full=False)
