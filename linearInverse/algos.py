import os

import numpy as np
import matplotlib.pyplot as plt
import regtools
import gurobipy as gp
import time


def algo1(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
          stepsize_inner, stepsize_bi, stepsize_dual, slater,
          initial_inner, initial_bi, initial_dual, n_iter, pre_iter=0, start_ave=-1):

    # initialization
    z0 = initial_inner
    x0 = initial_bi
    y0 = initial_dual

    # # FISTA
    # lmd0 = 0
    # eta = stepsize_inner
    # u0 = z0

    # mirror descent: record the *best* solution over the horizon
    min_z = z0
    fz0 = inner_obj(z0)

    # if start_ave < 0:
    #     start_ave = n_iter
    # w0 = 0*x0

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    # # mirror descent: first run a few iterations before generating the sequence needed
    # for i in range(pre_iter):
    #     z1 = proj(z0 - stepsize_inner * inner_obj_grad(z0))
    #     if inner_obj(z1) > inner_obj(z0):
    #         z1 = z0
    #     z0 = z1

    for i in range(n_iter):
        # # FISTA
        # lmd1 = .5 * ( 1 + np.sqrt(1+4*lmd0*lmd0) )
        # gamma = ( 1 - lmd0 ) / lmd1
        # u1 = proj( z0 - eta * inner_obj_grad(z0) )
        # z1 = (1-gamma) * u1 + gamma * u0

        # mirror descent: generate the pre-defined sequence
        z1 = proj( z0 - stepsize_inner * inner_obj_grad(z0) )
        fz1 = inner_obj(z1)
        if fz1 < fz0:
            min_z = z1

        # Algorithm 1: update rules
        dx = inner_obj_grad(x0)
        x1 = proj( x0 - stepsize_bi * ( outer_obj_grad(x0) + y0 * dx ) )
        y1 = max( 0, y0 + stepsize_dual *
                  ( inner_obj(x0) - inner_obj(min_z) - slater + np.dot(dx, x1-x0) ) )

        # update the variables
        z0 = z1
        x0 = x1
        y0 = y1
        fz0 = fz1
        # # FISTA
        # u0 = u1
        # lmd0 = lmd1

        # if i > start_ave:
        #     w0 = (w0*(i-start_ave-1) + x1)/(i-start_ave)
        # else:
        #     w0 = x0

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))
        # iters_inner.append(inner_obj(w0))
        # iters_outer.append(outer_obj(w0))

        # if ((i+1) % (n_iter // 100) == 0 or i < 100):
        #     print(f'{i+1:5}-th iteration, '
        #           f'inner est. = {inner_obj(z1):10.6f}, '
        #           f'inner obj. = {inner_obj(x0):10.6f}, '
        #           f'outer est. = {outer_obj(z1):8.4f}, '
        #           f'outer obj. = {outer_obj(x0):8.4f}, '
        #           f'lambda = {y1:8.2f}, '
        #           f'stepsize_x = {stepsize_bi:10.6f}, '
        #           f'stepsize_lambda = {stepsize_dual:10.6f}')

    return iters_inner, iters_outer


def algo2(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
          stepsize_inner, stepsize_bi, slater,
          initial_inner, initial_bi, initial_dual, n_iter, pre_iter=0, start_ave=-1):

    # initialization
    z0 = initial_inner
    x0 = initial_bi
    y0 = initial_dual
    g0 = -slater

    # mirror descent: record the *best* solution over the horizon
    min_z = z0
    fz0 = inner_obj(z0)

    # if start_ave < 0:
    #     start_ave = n_iter
    # w0 = 0*x0

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    # for i in range(pre_iter):
    #     z1 = proj(z0 - stepsize_inner * inner_obj_grad(z0))
    #     if inner_obj(z1) > inner_obj(z0):
    #         z1 = z0
    #     z0 = z1

    for i in range(n_iter):

        # mirror descent: generate the pre-defined sequence
        z1 = proj(z0 - stepsize_inner * inner_obj_grad(z0))
        fz1 = inner_obj(z1)
        if fz1 < fz0:
            min_z = z1

        # Algorithm 2: update rules
        x1 = proj(x0 - stepsize_bi * (outer_obj_grad(x0) + y0 * inner_obj_grad(x0)))
        g1 = inner_obj(x1) - inner_obj(min_z) - slater
        y1 = max(0, y0 + 2*g1 - g0)

        # update the variables
        z0 = z1
        x0 = x1
        y0 = y1
        g0 = g1
        fz0 = fz1

        # if i > start_ave:
        #     w0 = (w0*(i-start_ave-1) + x1)/(i-start_ave)
        # else:
        #     w0 = x0

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))
        # iters_inner.append(inner_obj(w0))
        # iters_outer.append(outer_obj(w0))

    return iters_inner, iters_outer


def algo1bt(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
            slater,
            initial_inner, initial_bi, initial_dual, n_iter):

    # iterate initialization
    z0 = initial_inner
    x0 = initial_bi
    y0 = initial_dual

    # backtracking parameter initialization
    bt = 1.1
    sy = 1
    lz, ez = 1 / 100, 1 + 1 / 32
    lx, ex = 1, 1 + 1 / 32

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    for i in range(n_iter):

        # backtracking mirror descent: generate the pre-defined sequence
        dz = inner_obj_grad(z0)
        while True:
            z1 = proj(z0 - dz / lz)
            if inner_obj(z1) - inner_obj(z0) \
                    <= np.vdot(dz, z1 - z0) + lz / 2 * np.vdot(z1 - z0, z1 - z0):
                break
            lz *= ez

        # backtracking Algorithm 1: primal update
        dg = inner_obj_grad(x0)
        dx = outer_obj_grad(x0) + y0 * dg
        while True:
            x1 = proj(x0 - dx / lx)
            if outer_obj(x1) - outer_obj(x0) \
                    + y0 * (inner_obj(x1) - inner_obj(x0)) \
                    <= np.vdot(dx, x1 - x0) + lx / 2 * np.vdot(x1 - x0, x1 - x0) + 1e-6:
                break
            lx *= ex
        x1 = proj(x0 - dx / lx / bt)
        # backtracking Algorithm 1: dual update
        dy = inner_obj(x0) - inner_obj(z1) - slater + np.vdot(dg, x1 - x0)
        y1 = np.maximum(0, y0 + sy * dy)

        # update the variables
        z0 = z1
        x0 = x1
        y0 = y1

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))

    return iters_inner, iters_outer


def algo2bt(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
            slater, h,
            initial_inner, initial_bi, initial_dual, n_iter):

    # iterate initialization
    z0 = initial_inner
    x0 = initial_bi
    y0 = initial_dual
    g0 = -slater

    # parameter initialization
    bt = 1.1
    lz, ez = 1 / 100, 1 + 1 / 32
    lx, ex = 1, 1 + 1 / 32

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    for i in range(n_iter):

        # backtracking mirror descent: generate the pre-defined sequence
        dz = inner_obj_grad(z0)
        while True:
            z1 = proj(z0 - dz / lz)
            if inner_obj(z1) - inner_obj(z0) \
                    <= np.vdot(dz, z1 - z0) + lz / 2 * np.vdot(z1 - z0, z1 - z0):
                break
            lz *= ez

        # backtracking Algorithm 2: primal update
        dx = outer_obj_grad(x0) + y0 * inner_obj_grad(x0)
        while True:
            x1 = proj(x0 - dx / lx)
            if (h * y0 - lx) * .5 * np.vdot(x1 - x0, x1 - x0) \
                    + 2 * (inner_obj(x1) - inner_obj(x0)) ** 2 < 1e-6:
                break
            lx *= ex
        x1 = proj(x0 - dx / lx / bt)
        # backtracking Algorithm 1: dual update
        g1 = inner_obj(x1) - inner_obj(z1) - slater
        y1 = max(0, y0 + 2*g1 - g0)

        # update the variables
        z0 = z1
        x0 = x1
        y0 = y1
        g0 = g1

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))

    return iters_inner, iters_outer


def solodov(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
            alpha, theta, eta,
            initial, n_iter):

    # initialization
    x0 = initial

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    for i in range(n_iter):
        # select sigma_i = 1/i
        phi = outer_obj(x0) / (i+1) + inner_obj(x0)
        phi_grad = outer_obj_grad(x0) / (i+1) + inner_obj_grad(x0)
        m = 0
        while True:
            x2 = proj(x0 - alpha * phi_grad)
            phi2 = outer_obj(x2) / (i+1) + inner_obj(x2)
            if phi2 <= phi + theta * np.dot(phi_grad, x2-x0) + 1e-10:
                break
            else:
                alpha *= eta
                m += 1
            # print(f'{i+1:5}-th iteration, '
            #       f'alpha = {alpha:10f}, '
            #       f'm = {m}, '
            #       f'stopping condition = {phi2 -( phi + theta * np.dot(phi_grad, x2-x0))}')
        x1 = proj(x0 - alpha * phi_grad)
        x0 = x1

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))

    return iters_inner, iters_outer


def BiGSAM(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
           t, s, alpha,
           initial, n_iter):

    # initialization
    x0 = initial

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    for i in range(n_iter):
        y1 = proj(x0 - t * inner_obj_grad(x0))
        z1 = x0 - s * outer_obj_grad(x0)
        x1 = alpha/(i+1) * z1 + (1-alpha/(i+1)) * y1
        x0 = x1

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))

    return iters_inner, iters_outer


def IRIG(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
         gamma, lbd, r, eps,
         initial, n_iter):

    # iterate initialization
    x0 = initial
    w0 = x0
    s0 = gamma ** r

    # parameter initialization
    gamma1 = gamma
    lbd1 = lbd

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    for i in range(n_iter):
        x1 = proj( x0 - gamma1 * ( inner_obj_grad(x0) + lbd1 * outer_obj_grad(x0) ) )

        gamma1 = gamma/(i+2)**(0.5+0.5*eps)
        lbd1 = lbd/(i+2)**(0.5-eps)

        s1 = s0 + gamma1 ** r
        w0 = (s0*w0 + (gamma1 ** r) * x1) / s1

        s0 = s1
        x0 = x1

        # record the iterations for plotting
        iters_inner.append(inner_obj(w0))
        iters_outer.append(outer_obj(w0))

    return iters_inner, iters_outer


def helou(inner_obj, inner_obj_grad, outer_obj, outer_obj_grad, proj,
          initial, n_iter):

    # iterate initialization
    x0 = initial

    # parameter initialization
    lmd = 0.05
    mu = 1
    y1 = x0 - lmd * inner_obj_grad(x0)
    z1 = x0 - mu * outer_obj_grad(y1)
    x1 = proj(z1)
    mu = np.linalg.norm(x0-y1) / np.linalg.norm(y1-z1) * 0.01

    # record the data of interest
    iters_inner = [inner_obj(x0)]
    iters_outer = [outer_obj(x0)]

    for i in range(n_iter):

        y1 = x0 - lmd / np.power(i+1, 0.1) * inner_obj_grad(x0)
        z1 = y1 - mu / (i+1) * outer_obj_grad(y1)
        x0 = proj(z1)

        # if ((i+1) % (n_iter // 100) == 0 or i < 100):
        #     print(f'{i+1:5}-th iteration, '
        #           f'inner est. = {inner_obj(x0):10.6f}, '
        #           f'outer est. = {outer_obj(x0):8.4f}, ')

        # record the iterations for plotting
        iters_inner.append(inner_obj(x0))
        iters_outer.append(outer_obj(x0))

    return iters_inner, iters_outer


def experiment(instance, bounded, n_iter):

    # problem setup
    n = 1000                # dimension
    rad = 20                # radius of bounded domain
    # if instance.__name__ == 'foxgood':
    #     h = .66
    #     lip = 1.5
    # elif instance.__name__ == 'baart':
    #     h = 11
    #     lip = .1
    # elif instance.__name__ == 'phillips':
    #     h = 35
    #     lip = .03

    # inner objective
    a, b, _ = instance(n)
    np.random.seed(1)
    b = np.reshape(b, -1) + np.random.randn(n) * 1e-2
    def sq_resid(x): return .5 * np.dot(a @ x - b, a @ x - b)
    def sq_resid_grad(x): return a.T @ (a @ x - b)

    # outer objective
    q = regtools.get_l(n+1, 1)
    q = q @ q.T + np.eye(n)
    def quad_form(x): return .5 * (x @ q @ x)
    def quad_form_grad(x): return q @ x

    def proj_l2ball(x):
        if np.dot(x, x) <= rad**2:
            return x
        else:
            return x / np.linalg.norm(x) * rad

    def proj_nonneg(x): return np.maximum(x, 0)

    # projection
    if bounded:
        proj = proj_l2ball
    else:
        proj = proj_nonneg

    # # solving the inner problem by gurobipy
    # print('Solving by Gurobi to get optimal inner value')
    # prob_inner = gp.Model()
    # if bounded:
    #     x = prob_inner.addMVar(n, lb=-gp.GRB.INFINITY)
    #     y = prob_inner.addMVar(n, lb=-gp.GRB.INFINITY)
    #     prob_inner.setObjective(y @ y * .5, gp.GRB.MINIMIZE)
    #     prob_inner.addConstr(a @ x - b == y)
    #     prob_inner.addConstr(x @ x <= rad**2)
    # else:
    #     x = prob_inner.addMVar(n)
    #     y = prob_inner.addMVar(n, lb=-gp.GRB.INFINITY)
    #     prob_inner.setObjective(y @ y * .5, gp.GRB.MINIMIZE)
    #     prob_inner.addConstr(a @ x - b == y)
    # prob_inner.update()
    # prob_inner.optimize()
    # opt_inner = prob_inner.objVal
    # print('')
    opt_inner = 0

    # theoretical stepsize gamma for algorithm 2
    norm_q = np.linalg.norm(q, 2)
    norm_a = np.linalg.norm(a, 2)
    norm_b = np.linalg.norm(b, 2)
    slater = np.power(n_iter, -1/2)
    Ff = .5 * norm_q * rad**2
    Fg = (norm_a * rad + norm_b)**2 + slater
    F = max(Ff, Fg)
    Gf = norm_q * rad
    Gg = norm_a * (norm_a * rad + norm_b)
    G = max(Gf, Gg)
    Hf = norm_q
    Hg = np.linalg.norm(a @ a.T, 2)
    Uf = n_iter * Ff
    Ug = Hg**2 / (4*n_iter) * rad**4
    M = Ff
    F0 = Ff
    gamma_algo2 = 1 / (4*Hg * (np.sqrt(2*Uf) + np.sqrt(2*Ug) + G*np.sqrt(2*n_iter) + 2*M/slater + 2*F + np.sqrt(2*F0))
                       + 4 * Hg**2 * (2*rad)**2 + 8 * G**2)

    time_list = []
    # solving the linear inverse problem
    if bounded:
        print('Running Algorithm 1...')
        tic = time.perf_counter()
        iters_inner_1, iters_outer_1 = \
            algo1(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
                  stepsize_inner=1/Gg, stepsize_bi=np.power(n_iter,-2/3), stepsize_dual=np.power(n_iter,-1/3),
                  slater=np.power(n_iter,-1/3),
                  initial_inner=np.zeros(n), initial_bi=np.zeros(n), initial_dual=0, n_iter=n_iter,
                  pre_iter=0, start_ave=-1)
        toc = time.perf_counter()
        print('- solution time (sec) = ' + f'{toc-tic}\n')
        time_list.append(toc - tic)

    print('Running Algorithm 1 with backtracking...')
    tic = time.perf_counter()
    iters_inner_1bt, iters_outer_1bt = \
        algo1bt(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
                slater=1e-4,
                initial_inner=np.zeros(n), initial_bi=np.zeros(n), initial_dual=100, n_iter=n_iter)
    toc = time.perf_counter()
    print('- solution time (sec) = ' + f'{toc-tic}\n')
    time_list.append(toc - tic)

    if bounded:
        print('Running Algorithm 2...')
        tic = time.perf_counter()
        iters_inner_2, iters_outer_2 = \
            algo2(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
                  stepsize_inner=1/Gg, stepsize_bi=gamma_algo2,
                  slater=np.power(n_iter,-1/2),
                  initial_inner=np.zeros(n), initial_bi=np.zeros(n), initial_dual=0, n_iter=n_iter,
                  pre_iter=0, start_ave=-1)
        toc = time.perf_counter()
        print('- solution time (sec) = ' + f'{toc-tic}\n')
        time_list.append(toc - tic)

    print('Running Algorithm 2 with backtracking...')
    tic = time.perf_counter()
    iters_inner_2bt, iters_outer_2bt = \
        algo2bt(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
                slater=1e-4, h=Hg,
                initial_inner=np.zeros(n), initial_bi=np.zeros(n), initial_dual=0, n_iter=n_iter)
    toc = time.perf_counter()
    print('- solution time (sec) = ' + f'{toc-tic}\n')
    time_list.append(toc - tic)

    print('Running BiG-SAM...')
    tic = time.perf_counter()
    iters_inner_bigsam, iters_outer_bigsam = \
        BiGSAM(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
               t=1/Hg, s=1/3, alpha=.6,
               initial=np.zeros(n), n_iter=n_iter)
    toc = time.perf_counter()
    print('- solution time (sec) = ' + f'{toc-tic}\n')
    time_list.append(toc - tic)

    print('Running IR-IG...')
    tic = time.perf_counter()
    iters_inner_irig, iters_outer_irig = \
        IRIG(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
             gamma=.1, lbd=.1, r=-10, eps=0,
             initial=np.zeros(n), n_iter=n_iter)
    toc = time.perf_counter()
    print('- solution time (sec) = ' + f'{toc-tic}\n')
    time_list.append(toc - tic)

    print('Running Solodov\'s algrithm...')
    tic = time.perf_counter()
    iters_inner_solodov, iters_outer_solodov = \
        solodov(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
                alpha=1, theta=.5, eta=.5,
                initial=np.zeros(n), n_iter=n_iter)
    toc = time.perf_counter()
    print('- solution time (sec) = ' + f'{toc-tic}\n')
    time_list.append(toc - tic)

    print('Running Helou\'s algrithm...')
    tic = time.perf_counter()
    iters_inner_helou, iters_outer_helou = \
        helou(sq_resid, sq_resid_grad, quad_form, quad_form_grad, proj,
              initial=np.zeros(n), n_iter=n_iter)
    toc = time.perf_counter()
    print('- solution time (sec) = ' + f'{toc-tic}\n')
    time_list.append(toc - tic)

    # save data
    iters_inner_list = [
        # iters_inner_1,
        iters_inner_1bt,
        # iters_inner_2,
        iters_inner_2bt,
        iters_inner_bigsam,
        iters_inner_irig,
        iters_inner_solodov,
        iters_inner_helou
    ]
    iters_outer_list = [
        # iters_outer_1,
        iters_outer_1bt,
        # iters_outer_2,
        iters_outer_2bt,
        iters_outer_bigsam,
        iters_outer_irig,
        iters_outer_solodov,
        iters_outer_helou
    ]
    name_list = ['algo1bt', 'algo2bt', 'bigsam', 'irig', 'solodov', 'helou']
    if bounded:
        iters_inner_list = [iters_inner_1, iters_inner_2] + iters_inner_list
        iters_outer_list = [iters_outer_1, iters_outer_2] + iters_outer_list
        name_list = ['algo1', 'algo2'] + name_list
        str_bdd = 'bounded_'
    else:
        str_bdd = 'unbounded_'
    # running time of all algorithms
    np.savetxt('output/' + str_bdd + instance.__name__ + '_time.csv', np.asarray([time_list]), header=','.join(name_list))
    # data of interest over the iterations for each algorithm
    for iters_inner, iters_outer, name in zip(iters_inner_list, iters_outer_list, name_list):
        tab = [[i+1, x-opt_inner] for i, x in enumerate(iters_inner)]
        np.savetxt('output/' + str_bdd + instance.__name__ + '_inner_' + name + '.csv', np.asarray(tab))
        tab = [[i+1, x] for i, x in enumerate(iters_outer)]
        np.savetxt('output/' + str_bdd + instance.__name__ + '_outer_' + name + '.csv', np.asarray(tab))

    # plots for inner objective
    plt.rc('text', usetex=True)
    if bounded:
        plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_1) - opt_inner, label='Algo1',  linestyle='-', color='C0')
    plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_1bt) - opt_inner, label='Algo1(backtracking)', linestyle='dashed', color='C0')
    if bounded:
        plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_2) - opt_inner, label='Algo2', linestyle='-', color='C1')
    plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_2bt) - opt_inner, label='Algo2(backtracking)', linestyle='dashed', color='C1')
    plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_bigsam) - opt_inner, label='BiG-SAM', color='C2')
    plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_irig) - opt_inner, label='IR-IG', color='C3')
    plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_solodov) - opt_inner, label='Solodov', color='C4')
    plt.loglog(np.arange(1, n_iter+2), np.array(iters_inner_helou) - opt_inner, label='Helou', color='C5')
    plt.title(instance.__name__ + (r', $X=\{\|x\|_2\leq\,$' + f'{rad:2}' + r'$\}$' if bounded else r', $X=R_+$'))
    plt.ylabel(r'$\eta(x_t)-\eta^*$')
    plt.xlabel(r'iteration $t$')
    plt.legend()
    plt.show()

    # plots for outer objectives
    if bounded:
        plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_1), label='Algo1',  linestyle='-', color='C0')
    plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_1bt), label='Algo1(backtracking)', linestyle='dashed', color='C0')
    if bounded:
        plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_2), label='Algo2', linestyle='-', color='C1')
    plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_2bt), label='Algo2(backtracking)', linestyle='dashed', color='C1')
    plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_bigsam), label='BiG-SAM', color='C2')
    plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_irig), label='IR-IG', color='C3')
    plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_solodov), label='Solodov', color='C4')
    plt.plot(np.arange(1, n_iter+2), np.array(iters_outer_helou), label='Helou', color='C5')
    plt.title(instance.__name__ + (r', $X=\{\|x\|_2\leq\,$' + f'{rad:2}' + r'$\}$' if bounded else r', $X=R_+$'))
    plt.ylabel(r'$\phi(x_t)$')
    plt.xlabel(r'iteration $t$')
    plt.legend()
    plt.show()

    if bounded:
        return time_list
    else:
        return [0, 0] + time_list


def subsampling_inner(instance, bounded, n_sample=400):
    # subsampling iterates for plots
    str_bdd = 'bounded_' if bounded else 'unbounded_'
    name_list = ['algo1bt', 'algo2bt', 'bigsam', 'irig', 'solodov', 'helou']
    if bounded:
        name_list = ['algo1', 'algo2'] + name_list
    for name in name_list:
        tab = np.loadtxt('output/' + str_bdd + instance.__name__ + '_inner_' + name + '.csv')
        subtab = [tab[0]]
        for i in range(n_sample+1):
            ind = np.floor(np.power(n_iter, i/n_sample)).astype(int)
            if subtab[-1][0] < ind:
                subtab.append(tab[ind])
        np.savetxt('output/' + str_bdd + instance.__name__ + '_inner_' + name + '_subsampled.csv', np.asarray(subtab))


def subsampling_outer(instance, bounded, n_sample=400):
    str_bdd = 'bounded_' if bounded else 'unbounded_'
    name_list = ['algo1bt', 'algo2bt', 'bigsam', 'irig', 'solodov', 'helou']
    if bounded:
        name_list = ['algo1', 'algo2'] + name_list
    for name in name_list:
        tab = np.loadtxt('output/' + str_bdd + instance.__name__ + '_outer_' + name + '.csv')
        subtab = []
        for i in range(n_sample+1):
            ind = np.floor(n_iter/n_sample*i).astype(int)
            subtab.append([ind, tab[ind][1]])
        np.savetxt('output/' + str_bdd + instance.__name__ + '_outer_' + name + '_subsampled.csv', np.asarray(subtab))


if __name__ == '__main__':
    n_iter = 10000
    timing = []
    os.makedirs('output', exist_ok=True)
    for bounded in [True]:  #, False]:
        for instance in [regtools.foxgood]:  #, regtools.baart, regtools.phillips]:
            print('--------------------------------------------')
            print('Solving ' + instance.__name__ + ', ' + ('bounded' if bounded else 'unbounded') + ' case: \n')
            timing.append(experiment(instance, bounded, n_iter))
    print(np.asarray(timing).T)
    np.savetxt('output/linear_inverse_time.csv', np.asarray(timing),
               header=','.join(['algo1', 'algo2', 'algo1bt', 'algo2bt', 'bigsam', 'irig', 'solodov', 'helou']))
