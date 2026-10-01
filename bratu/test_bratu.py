import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.linalg import eigvalsh_tridiagonal

import pickle

import time

Nx = 200
Ny = 200
h1 = 1./(Nx+1)
h2 = 1./(Ny+1)
alpha = 20.
lamb = 1.


X = np.linspace(0,1,Nx+2)
Y = np.linspace(0,1,Ny+2)

#U_global = np.zeros((Nx+2,Ny+2))

def norm2(x):
    return np.linalg.norm(x,ord=2)

from numpy.linalg import pinv


def f(U_global):
    Ul,Ur = U_global[0:Nx,1:Ny+1],U_global[2:Nx+2,1:Ny+1]
    U = U_global[1:Nx+1,1:Ny+1]
    U_tmp1 = (Ur-2*U+Ul)/(h1**2)
    
    Ud,Uu = U_global[1:Nx+1,0:Ny],U_global[1:Nx+1,2:Ny+2]
    U_tmp2 = (Uu-2*U+Ud)/(h2**2)
    
    U_tmp3 = alpha*(Ur-Ul)/(2*h1)
    
    U_tmp4 = lamb*np.exp(U)
    
    fU = U_tmp1+U_tmp2+U_tmp3+U_tmp4 
    
    return norm2(fU),fU

'''def matvec(U_star,U_global):
    Ul, Ur = U_global[0:Nx, 1:Ny + 1], U_global[2:Nx + 2, 1:Ny + 1]
    U = U_global[1:Nx + 1, 1:Ny + 1]
    U_tmp1 = (Ur - 2 * U + Ul) / (h1 ** 2)

    Ud, Uu = U_global[1:Nx + 1, 0:Ny], U_global[1:Nx + 1, 2:Ny + 2]
    U_tmp2 = (Uu - 2 * U + Ud) / (h2 ** 2)

    U_tmp3 = alpha * (Ur - Ul) / (2 * h1)

    U_tmp4 = lamb * np.exp(U_star[1:Nx+1,1:Ny+1])
    U_tmp4 *= U

    fU = U_tmp1 + U_tmp2 + U_tmp3 + U_tmp4

    return norm2(fU), -fU '''

def matvec(U_star,v):
    epsilon = 1e-8
    fk,rk = f(U_star)
    U_next = U_star+epsilon*v
    fk2,rk2 = f(U_next)
    Av = -(rk2-rk)/epsilon
    return norm2(Av), Av
    

def arnoldi(f, U_star, U0, maxit=100):
    cnt = 0
    Uk = U0.copy()
    fk, rk = f(Uk)

    beta = norm2(rk)
    Vk = [rk / beta]
    H = []
    flag = False
    while cnt < maxit and (not flag):
        v = Vk[-1]
        #w = A @ v
        Uk[1:Nx+1,1:Ny+1] = v
        _, w = matvec(U_star,Uk)
        hs = []
        for v in Vk:
            h = np.vdot(w.reshape(-1), v.reshape(-1))
            w -= h * v
            hs.append(h)
        h = norm2(w.reshape(-1))
        hs.append(h)
        H.append(hs)
        if h == 0:
            m = cnt + 1
            flag = True
        else:
            Vk.append(w / h)

        cnt = cnt + 1

    Hk_bar = np.zeros((len(H) + 1, len(H)))
    for j in range(0, len(H)):
        Hk_bar[0:len(H[j]), j] = H[j]
    Hk = Hk_bar[0:len(H), 0:len(H)]

    #print(Hk)

    # print(Hk_bar)
    eig = np.linalg.eigvals(Hk)

    # import matplotlib.pyplot as plt
    # plt.scatter(eig.real, eig.imag, label='eigenvalue', color='red', marker='o')
    # if eig_estimate is not None:
    #     plt.scatter(eig_estimate.real, eig_estimate.imag, label='AM Ritz value', color='blue', marker='+')
    # plt.legend()
    # plt.show()

    print("Arnoldi method for eigenvalue computation.")
    return eig


def picard(f,U0,tol=1e-8,maxit=1e5,beta=1e-4):
    cnt = 0
    #res_v = []
    Uk = U0
    f_val, fU = f(Uk)
    res_v = []   
    time_v = []
    start = time.time()
    while cnt<maxit and f_val>tol:
        print(cnt,f_val)
        res_v.append(f_val)
        time_v.append(time.time()-start)
        Uk[1:Nx+1,1:Ny+1] += beta*fU
        cnt = cnt+1
        f_val, fU = f(Uk)
    
    res_v.append(f_val)
    time_v.append(time.time()-start)
    pickle.dump((res_v,time_v),open('picard.ser','wb'))
    return Uk
        

def anderson2(f,U0,tol=1e-8,maxit=1e5,beta=1e-4,m=20):
    cnt = 0
    Uk = U0
    fk,res = f(Uk)
    
    xk = Uk[1:Nx+1,1:Ny+1]
    N = Nx*Ny
    Xk = np.zeros((N,m))
    Rk = np.zeros((N,m))
    mk = 0
    res_v = []   
    time_v = []
    start = time.time()
    while cnt<maxit and fk>tol:
        print(cnt,fk)
        res_v.append(fk)
        time_v.append(time.time()-start)

        if cnt >= 1:
            if cnt <= m:
                mk = cnt
                Xk[:,mk-1] = (xk-x_prev).reshape(-1)
                Rk[:,mk-1] = (res-res_prev).reshape(-1)
            else:
                mk = m
                Xk[:,:-1],Xk[:,-1] = Xk[:,1:],(xk-x_prev).reshape(-1)
                Rk[:,:-1],Rk[:,-1] = Rk[:,1:],(res-res_prev).reshape(-1)
            
        x_prev = xk.copy()
        res_prev = res.copy()

        Xk_tmp, Rk_tmp = Xk[:,0:mk],Rk[:,0:mk]
        #Gamma = pinv(Rk_tmp.T@Rk_tmp)@(Rk_tmp.T@res.reshape(N,1))
        Gamma = np.linalg.solve(Rk_tmp.T@Rk_tmp,Rk_tmp.T@res.reshape(N,1))
        xk_bar = xk-(Xk_tmp @ Gamma).reshape(Nx,Ny)
        rk_bar = res-(Rk_tmp @ Gamma).reshape(Nx,Ny)
        xk = xk_bar + beta*rk_bar  
        Uk[1:Nx+1,1:Ny+1] = xk        
        
        cnt = cnt+1
        fk,res = f(Uk)
    
    res_v.append(fk)
    time_v.append(time.time()-start)
    pickle.dump((res_v,time_v),open('anderson2_'+str(m)+'.ser','wb'))
    #print('Anderson method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (relres,fk,cnt))
    return Uk


def anderson1(f, U0, tol=1e-8, maxit=1e5, beta=1e-4, m=20):
    cnt = 0
    Uk = U0
    fk, res = f(Uk)

    xk = Uk[1:Nx + 1, 1:Ny + 1]
    N = Nx * Ny
    Xk = np.zeros((N, m))
    Rk = np.zeros((N, m))
    mk = 0
    res_v = []
    time_v = []
    start = time.time()
    while cnt < maxit and fk > tol:
        print(cnt, fk)
        res_v.append(fk)
        time_v.append(time.time() - start)

        if cnt >= 1:
            if cnt <= m:
                mk = cnt
                Xk[:, mk - 1] = (xk - x_prev).reshape(-1)
                Rk[:, mk - 1] = (res - res_prev).reshape(-1)
            else:
                mk = m
                Xk[:, :-1], Xk[:, -1] = Xk[:, 1:], (xk - x_prev).reshape(-1)
                Rk[:, :-1], Rk[:, -1] = Rk[:, 1:], (res - res_prev).reshape(-1)

        x_prev = xk.copy()
        res_prev = res.copy()

        Xk_tmp, Rk_tmp = Xk[:, 0:mk], Rk[:, 0:mk]
        #Gamma = pinv(Xk_tmp.T @ Rk_tmp) @ (Xk_tmp.T @ res.reshape(N, 1))
        Gamma = np.linalg.solve(Xk_tmp.T @ Rk_tmp,Xk_tmp.T @ res.reshape(N, 1))
        xk_bar = xk - (Xk_tmp @ Gamma).reshape(Nx, Ny)
        rk_bar = res - (Rk_tmp @ Gamma).reshape(Nx, Ny)
        xk = xk_bar + beta * rk_bar
        Uk[1:Nx + 1, 1:Ny + 1] = xk

        cnt = cnt + 1
        fk, res = f(Uk)

    res_v.append(fk)
    time_v.append(time.time()-start)
    pickle.dump((res_v,time_v), open('anderson1_' + str(m) + '.ser', 'wb'))
    # print('Anderson method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (relres,fk,cnt))
    return Uk
   


def am2(f, U0, tol=1e-8, maxit=1e5, tao=1e-16, beta=1.0, m=100):
    cnt = 0
    Uk = U0
    fk, rk = f(Uk)

    xk = Uk[1:Nx + 1, 1:Ny + 1]
    N = Nx*Ny
    Pk = np.zeros((N, m))
    Qk = np.zeros((N, m))

    Hk_bar = np.zeros((m + 1, m))
    zetas = np.zeros(m)
    gammas = np.zeros(m)
    xis = np.zeros(m)
    #cur = -1
    #length = 0
    mk = -1

    qqs = []

    beta_prev = beta

    res_v = []
    time_v = []
    beta_v = []
    # restart = False
    start = time.time()
    while cnt < maxit and fk > tol:
        print(cnt,fk)
        res_v.append(fk)
        time_v.append(time.time() - start)

        if mk >= 0:
            restart = False
            p = (xk - x_prev).reshape(-1)
            q = (rk - r_prev).reshape(-1)
            for j in range(0, mk):
                v = Qk[:, j]
                zeta = np.vdot(q, v) / np.vdot(v, v)
                q -= zeta * v
                p -= zeta * Pk[:, j]
                zetas[j] = zeta

            # q = -A@p                   #critical
            # print('$',norm2(q+A@p))

            qq = np.vdot(q, q)
            qqs.append(qq)
            if qqs[-1] <= (tao) * qqs[0] or mk == m:
                restart = True
                mk = -1
                qqs = []
                beta_prev = beta
                print('restart!')

            if not restart:
                Pk[:, mk] = p
                Qk[:, mk] = q

            if mk >= 1 and mk<=0:
                # print(zetas[0:length-1],gammas[0:length-1])
                # assert len(zetas) == len(gammas)
                xis[0:mk] = gammas[0:mk] + zetas[0:mk]
                xis[mk] = 1.
                h = Hk_bar[:, mk-1]
                h.fill(0)
                if mk == 1:
                    # assert len(Pk) == 2
                    h[0] = 1. / beta_prev - 1. / beta * xis[0]
                    h[1] = -1. / beta
                    h[0:mk+1] /= 1 - gammas[0]
                else:
                    h[0:mk] = 1. / beta_prev * xis_prev - 1. / beta * xis[0:mk] - (
                                Hk_bar[0:mk, 0:mk-1] @ (xis_prev[0:-1] - gammas[0:mk-1]).reshape(
                            mk-1, 1)).reshape(-1)
                    h[mk] = -1. / beta
                    h[0:mk+1] /= 1 - gammas[mk-1]

                Hk = Hk_bar[0:mk, 0:mk]

                # print(Hk_bar)
                eig = np.linalg.eigvals(Hk)
                # print(eig)
                # print(Hk)
                beta_prev = beta
                #beta = 1. / abs(max(eig))
                beta = 2./max(abs(eig))
                #print('beta:',beta)

                """Q_tmp = Qk[:, 0:mk]
                P_tmp = Pk[:, 0:mk]

                H_tmp = np.linalg.inv(Q_tmp.T @ Q_tmp) @ (Q_tmp.T @ (A @ Q_tmp))
                # for j in range(0,H_tmp.shape[1]):
                #    H_tmp[j+2:,j].fill(0)
                # print(H_tmp[-1,:])
                print('*', np.linalg.norm(Hk - H_tmp) / np.linalg.norm(H_tmp))
                # eig = np.linalg.eigvals(H_tmp)
                # beta_prev = beta
                # beta = 1./abs(max(eig))
                # print(eig)
                # print(H_tmp)
                # print(np.linalg.norm(Hk),np.linalg.norm(H_tmp))"""


        x_prev = xk.copy()
        r_prev = rk.copy()
        xis_prev = xis[0:mk+1].copy()

        mk = mk+1
        for j in range(0, mk):
            p, q = Pk[:, j], Qk[:, j]
            gamma = np.vdot(rk, q) / np.vdot(q, q)
            rk -= gamma * q.reshape(rk.shape)
            xk -= gamma * p.reshape(xk.shape)
            gammas[j] = gamma

        xk += beta * rk  # rk_bar = rk
        Uk[1:Nx + 1, 1:Ny + 1] = xk
        cnt = cnt + 1
        beta_v.append(beta)
        fk, rk = f(Uk)

    #fk, gk = fh(xk)
    res_v.append(fk)
    time_v.append(time.time()-start)
    pickle.dump(res_v, open('am2_' + str(m) + '.ser', 'wb'))

    #eig_tmp = np.linalg.eigvals(A)
    '''eig_tmp = arnoldi(f,Uk.copy(),np.zeros(U0.shape),mk)
    print('****************')
    # print(eig_tmp)

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['text.usetex'] = True
    print(len(eig_tmp), len(eig))    
    plt.tick_params(labelsize='large')
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='Ritz value', color='darkorange', marker='o')
    plt.scatter(eig.real, eig.imag, label='AM2', color='blue', marker='x')
    plt.legend(fontsize='large')
    #plt.show()
    plt.savefig('figure_am2_bratu.pdf',format='pdf',dpi=120,bbox_inches='tight')
    pickle.dump((eig,eig_tmp,beta_v), open('am2_eig_beta_' + str(mk) + '.ser', 'wb')) '''
    #print('am2 method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (
    #        norm2(gk) / nd, fk, cnt))
    return Uk

def am1(f, U0, tol=1e-8, maxit=1e5, tao=1e-16, beta=1.0, m=100):
    cnt = 0
    Uk = U0
    fk, rk = f(Uk)

    xk = Uk[1:Nx + 1, 1:Ny + 1]
    N = Nx * Ny
    Pk = np.zeros((N, m))
    Qk = np.zeros((N, m))

    Hk_bar = np.zeros((m + 1, m))
    zetas = np.zeros(m)
    gammas = np.zeros(m)
    xis = np.zeros(m)
    # cur = -1
    # length = 0
    mk = -1

    pqs = []

    beta_prev = beta

    res_v = []
    time_v = []
    beta_v = []
    # restart = False
    start = time.time()
    while cnt < maxit and fk > tol:
        print(cnt, fk)
        res_v.append(fk)
        time_v.append(time.time() - start)

        if mk >= 0:
            restart = False
            p = (xk - x_prev).reshape(-1)
            q = (rk - r_prev).reshape(-1)
            for j in range(0, mk):
                pj,qj = Pk[:,j],Qk[:,j]
                zeta = np.vdot(q, pj) / np.vdot(pj, qj)
                q -= zeta * qj
                p -= zeta * pj
                zetas[j] = zeta

            # q = -A@p                   #critical
            # print('$',norm2(q+A@p))

            pq = abs(np.vdot(p, q))
            pqs.append(pq)
            if pqs[-1] <= (tao) * pqs[0] or mk == m:
                restart = True
                mk = -1
                pqs = []
                beta_prev = beta
                print('restart!')

            if not restart:
                Pk[:, mk] = p
                Qk[:, mk] = q

            if mk >= 1 and mk<=0:
                # print(zetas[0:length-1],gammas[0:length-1])
                # assert len(zetas) == len(gammas)
                xis[0:mk] = gammas[0:mk] + zetas[0:mk]
                xis[mk] = 1.
                h = Hk_bar[:, mk-1]
                h.fill(0)
                if mk == 1:
                    # assert len(Pk) == 2
                    h[0] = 1. / beta_prev - 1. / beta * xis[0]
                    h[1] = -1. / beta
                    h[0:mk+1] /= 1 - gammas[0]
                else:
                    h[0:mk] = 1. / beta_prev * xis_prev - 1. / beta * xis[0:mk] - (
                                Hk_bar[0:mk, 0:mk-1] @ (xis_prev[0:-1] - gammas[0:mk-1]).reshape(
                            mk-1, 1)).reshape(-1)
                    h[mk] = -1. / beta
                    h[0:mk+1] /= 1 - gammas[mk-1]

                #print("1-gammas[mk-1]:", 1-gammas[mk-1])

                Hk = Hk_bar[0:mk, 0:mk]

                # print(Hk_bar)
                eig = np.linalg.eigvals(Hk)
                # print(eig)
                # print(Hk)
                beta_prev = beta
                #beta = 1. / abs(max(eig))
                beta = 2. / max(abs(eig))

                """Q_tmp = Qk[:, 0:mk]
                P_tmp = Pk[:, 0:mk]

                H_tmp = np.linalg.inv(P_tmp.T @ Q_tmp) @ (P_tmp.T @ (A @ Q_tmp))
                # for j in range(0,H_tmp.shape[1]):
                #    H_tmp[j+2:,j].fill(0)
                # print(H_tmp[-1,:])
                print('*', np.linalg.norm(Hk - H_tmp) / np.linalg.norm(H_tmp))
                # eig = np.linalg.eigvals(H_tmp)
                # beta_prev = beta
                # beta = 1./abs(max(eig))
                # print(eig)
                # print(H_tmp)
                # print(np.linalg.norm(Hk),np.linalg.norm(H_tmp))"""


        x_prev = xk.copy()
        r_prev = rk.copy()
        xis_prev = xis[0:mk+1].copy()

        mk = mk+1
        for j in range(0, mk):
            p, q = Pk[:, j], Qk[:, j]
            gamma = np.vdot(rk, p) / np.vdot(p, q)
            rk -= gamma * q.reshape(rk.shape)
            xk -= gamma * p.reshape(xk.shape)
            gammas[j] = gamma

        xk += beta * rk  # rk_bar = rk
        Uk[1:Nx + 1, 1:Ny + 1] = xk
        cnt = cnt + 1
        beta_v.append(beta)
        fk, rk = f(Uk)

    res_v.append(fk)
    time_v.append(time.time()-start)
    pickle.dump(res_v, open('am1_' + str(m) + '.ser', 'wb'))

    #eig_tmp = np.linalg.eigvals(A)
    '''eig_tmp = arnoldi(f, Uk.copy(), np.zeros(U0.shape), mk-1)
    print('****************')
    # print(eig_tmp)

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['text.usetex'] = True
    print(len(eig_tmp), len(eig))    
    plt.tick_params(labelsize='large')
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='Ritz value', color='darkorange', marker='o')
    plt.scatter(eig.real, eig.imag, label='AM1', color='purple', marker='+')
    plt.legend(fontsize='large')
    #plt.show()
    plt.savefig('figure_am1_bratu.pdf',format='pdf',dpi=120,bbox_inches='tight')
    pickle.dump((eig,eig_tmp,beta_v), open('am1_eig_beta_' + str(mk) + '.ser', 'wb')) '''
    #print('am1 method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (
    #        norm2(gk) / nd, fk, cnt))
    return Uk


def lu_Crout(A):    
    n = A.shape[0]
    A = A.copy()
    for k in range(0,n-1):
        A[k,k+1:n] = A[k,k+1:n]/A[k,k]
        A[k+1:n,k+1:n] = A[k+1:n,k+1:n] - A[k+1:n,k:k+1]@A[k:k+1,k+1:n]   #            
    return A

def qr_single(H,a):
    n = H.shape[0]
    Q,R = np.linalg.qr(H-a*np.eye(n))
    H = R@Q+a*np.eye(n)        
    if n>1:
        H[n-1,n-2] = 0
    H -= np.tril(H,-2)
    return H,Q

def qr_double(H,a,b):
    n = H.shape[0]
    s = a*2
    t = a**2+b**2
    M = H@H-s*H+t*np.eye(n)
    Z,R = np.linalg.qr(M)
    H2 = Z.T@H@Z    
    if n>2:
        H2[n-2,n-3] = 0
    H2 -= np.tril(H2,-2)    
    return H2,Z     

def qr_filter(H,eigs):
    n = H.shape[0]    
    eig_list = []
    k = 0
    Z = np.eye(n)
    H0 = H.copy()
    while k<len(eigs):
        eig = eigs[k]        
        if np.isreal(eig):  
            H[0:n-k,0:n-k],Q1 = qr_single(H[0:n-k,0:n-k],np.real(eig))
            H[0:n-k,n-k:n] = Q1.T@H[0:n-k,n-k:n]
            Z[0:n,0:n-k] = Z[0:n,0:n-k]@Q1
            k = k+1            
        else:            
            H[0:n-k,0:n-k],Q2 = qr_double(H[0:n-k,0:n-k],np.real(eig),np.imag(eig))
            H[0:n-k,n-k:n] = Q2.T@H[0:n-k,n-k:n]
            Z[0:n,0:n-k] = Z[0:n,0:n-k]@Q2
            k = k+2                    
    return H,Z

def iram(f, U0, tol=1e-8, maxit=1e5, tao=1e-16, beta=1.0, m=10,n=2,mit=100, type2=True ,adaptive_beta=False):
    cnt = 0
    Uk = U0
    fk, rk = f(Uk)
    
    N = Nx * Ny
    xk = Uk[1:Nx + 1, 1:Ny + 1].reshape(N,1)
    rk = rk.reshape(N,1)
    Pk = np.zeros((N, m+1))
    Qk = np.zeros((N, m+1))
    Vk = Qk if type2 else Pk

    Hk_bar = np.zeros((m + 1, m))    
    mk = -1
    Mk = -1

    res_v = []    
    #relres_bar_v = []    
    # restart = False
    vqs = []
    time_v = []
    beta_v = []
    start = time.time()

    beta_prev = beta
    while cnt < maxit and fk > tol:
        print(cnt, fk)
        res_v.append(fk)
        time_v.append(time.time() - start)

        if mk >= 0:
            restart = False
            p = xk - x_prev
            q = rk - r_prev
            if mk>0:
                p = p+Pk[:,0:mk]@gammak
                q = q+Qk[:,0:mk]@gammak 
            Pk[:, mk] = p.reshape(-1)
            Qk[:, mk] = q.reshape(-1)            
            VQ = Vk[:,0:mk+1].T@Qk[:,0:mk+1]
            #vq = VQ[-1,-1]-VQ[mk:mk+1,0:mk]@np.linalg.solve(VQ[0:mk,0:mk],VQ[0:mk,mk:mk+1])
            VQ_lu = lu_Crout(VQ)
            vq = np.min(VQ_lu.diagonal())
            vqs.append(abs(vq))
            if vqs[-1]/vqs[0] <= tao or Mk == mit:
                restart = True
                mk = -1
                Mk = -1
                vqs = []
                beta_prev = beta
                print('restart!') 
                

            if mk >= 1:
                #xis[0:mk] = gammas[0:mk] + zetas[0:mk]
                #xis[mk] = 1.
                h = Hk_bar[:, mk-1]
                h.fill(0)
                if mk == 1:
                    h[0] = 1. / beta_prev 
                    h[1] = -1. / beta
                    h[0:mk+1] /= 1 - gammak[0]
                else:
                    h[0:mk] =  (Hk_bar[0:mk, 0:mk-1] @ gammak[0:mk-1,0:1]).reshape(-1)
                    h[mk-1] = h[mk-1]+1/beta_prev
                    h[mk] = -1. / beta
                    h[0:mk+1] /= 1 - gammak[mk-1]

                Hk = Hk_bar[0:mk, 0:mk]

                eig = np.linalg.eigvals(Hk)
                # 保存当前 beta，供下一次构造 Hk 使用
                beta_prev = beta

                if adaptive_beta and eig.size > 0:
                    largest_eigenvalue = np.max(np.abs(eig))

                    if (
                            np.isfinite(largest_eigenvalue)
                            and largest_eigenvalue > 0
                    ):
                        beta = 2.0 / largest_eigenvalue
                #beta = 1. / abs(max(eig))
                #beta = 2./ max(abs(eig))
                
                if mk == m:
                    eigs = np.sort_complex(eig)[::-1]
                    nk = n
                    if np.iscomplex(eigs[n-1]): # 0~n-1
                        nk = n
                        if n<m and np.conj(eigs[n]) == eigs[n-1]:
                            nk = n+1                        
                    Hk,Z = qr_filter(Hk,eigs[0:nk])       
                    #print(eigs)
                    '''print(nk)
                    print('H:',Hk)
                    print('Z:',Z) '''
                    Pk[:,0:m-nk] = (Pk[:,0:m]@Z)[:,0:m-nk]
                    Qk[:,0:m-nk] = (Qk[:,0:m]@Z)[:,0:m-nk]
                    Pk[:,m-nk:m+1].fill(0)
                    Pk[:,m-nk] = p.reshape(-1)
                    Qk[:,m-nk] = q.reshape(-1)
                    mk = m-nk     
                    if mk>0:                                   
                        Hk = Hk[0:mk,0:mk]
                        Hk_bar[0:mk,0:mk] = Hk
                        Hk_bar[mk,mk-1] = Hk_bar[m,m-1]*Z[m-1,mk-1]
                    
                        '''print('***************')     
                        P_tmp = Pk[:,0:mk]
                        V_tmp = Vk[:,0:mk] 
                        Q_tmp = Qk[:,0:mk]
                        #H_tmp = np.linalg.inv(V_tmp.T @ P_tmp) @ (V_tmp.T @ (A @ P_tmp))
                        H_tmp = np.linalg.inv(V_tmp.T @ P_tmp) @ (-V_tmp.T @ Q_tmp)
                        print('^', np.linalg.norm(Hk - H_tmp) / np.linalg.norm(H_tmp)) '''
                    VQ = Vk[:,0:mk+1].T@Qk[:,0:mk+1]                      
                
                '''if mk>0:
                    P_tmp = Pk[:,0:mk]
                    Q_tmp = Qk[:,0:mk]
                    V_tmp = Vk[:,0:mk] 

                    #H_tmp = np.linalg.inv(V_tmp.T @ P_tmp) @ (V_tmp.T @ (A @ P_tmp))
                    H_tmp = np.linalg.inv(V_tmp.T @ P_tmp) @ (-V_tmp.T @ Q_tmp)
                    # for j in range(0,H_tmp.shape[1]):
                    #    H_tmp[j+2:,j].fill(0)
                    # print(H_tmp[-1,:])
                    print('*', np.linalg.norm(Hk - H_tmp) / np.linalg.norm(H_tmp))  '''


        x_prev = xk.copy()
        r_prev = rk.copy()        

        mk = mk+1
        Mk = Mk+1
        
        if mk>0:            
            P_tmp = Pk[:,0:mk]
            Q_tmp = Qk[:,0:mk]
            V_tmp = Vk[:,0:mk]
            gammak = np.linalg.solve(VQ,V_tmp.T@rk) 
            #gammak = lu_solve(VQ_lu,V_tmp.T@rk) 
            xk = xk-P_tmp@gammak
            rk = rk-Q_tmp@gammak        
            
            '''            
            #gammak2 = np.linalg.solve(VQ,V_tmp.T@rk) 
            gammak2 = lu_solve(VQ_lu,V_tmp.T@rk) 
            xk = xk-P_tmp@gammak2
            rk = rk-Q_tmp@gammak2
            gammak = gammak+gammak2 '''
        

        xk += beta * rk  # rk_bar = rk
        Uk[1:Nx + 1, 1:Ny + 1] = xk.reshape(Nx,Ny)
        cnt = cnt + 1
        #beta_v.append(beta)
        if adaptive_beta:
            beta_v.append(float(beta))
        fk, rk = f(Uk)
        rk = rk.reshape(N,1)
        
    #fk, gk = fh(xk)
    res_v.append(fk)    
    time_v.append(time.time() - start)
    if type2:
        filename = 'iram2_'+str(m)+'_'+str(n)+'.ser'
    else:
        filename = 'iram1_'+str(m)+'_'+str(n)+'.ser'
    pickle.dump(
        (res_v, time_v),
        open(filename, "wb"),
    )

    if adaptive_beta:
        beta_filename = filename.replace(
            ".ser",
            "_beta.ser",
        )

        pickle.dump(
            beta_v,
            open(beta_filename, "wb"),
        )

    #eig_tmp = np.linalg.eigvals(A)
    eig_tmp = arnoldi(f, Uk.copy(), np.zeros(U0.shape), mk-1)
    print('****************')
    # print(eig_tmp)

    import matplotlib.pyplot as plt
    print(len(eig_tmp), len(eig))
    plt.clf()
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='red', marker='o')
    plt.scatter(eig.real, eig.imag, label='IRAM', color='blue', marker='+')
    plt.legend()
    plt.show()
    #plt.savefig('figure_am2_linear.pdf',format='pdf',dpi=120,bbox_inches='tight') 
    #print('iram method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (
    #        norm2(gk) / nd, fk, cnt))  
    return Uk
            

#U0 = np.zeros((Nx+2,Ny+2))

#Uk = picard(f,U0.copy(),tol=1e-6,maxit=1e5,beta=6e-6)
#Uk = picard(f,U0.copy(),tol=1e-6,maxit=1e5,beta=1e-7)
#Uk = anderson2(f,U0,tol=1e-6,maxit=1e5,beta=6e-6,m=26)
#Uk = anderson1(f,U0,tol=1e-6,maxit=1e5,beta=6e-6,m=26)
#Uk = anderson1(f,U0,tol=1e-6,maxit=1e5,beta=6e-6,m=1000)
#Uk = am1(f,U0.copy(),tol=1e-6,maxit=2000,tao=1e-32,beta=6e-6,m=26)
#Uk = am1(f,U0.copy(),tol=1e-6,maxit=2000,tao=0,beta=6e-6,m=50)
#Uk = am1(f,U0.copy(),tol=1e-6,maxit=1e5,tao=1e-32,beta=1.,m=1000)
#Uk = iram(f,U0.copy(),tol=1e-6,maxit=1e5,tao=1e-16,beta=6e-6,m=30,n=15,mit=1000,type2=True)
#Uk = iram(f,U0.copy(),tol=1e-6,maxit=1e5,tao=1e-16,beta=6e-6,m=25,n=15,mit=1000,type2=True)
#Uk = iram(f,U0.copy(),tol=1e-6,maxit=1e5,tao=1e-16,beta=6e-6,m=1000,n=16,mit=1000,type2=True)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#plt.grid()
#X,Y = np.meshgrid(X,Y)
#surf = ax.plot_surface(X,Y,Uk,cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f')) 
#fig.colorbar(surf, shrink=0.5, aspect=5)                      
#fig.colorbar(surf)
#plt.show()
#plt.savefig('bratu_nonsym.pdf',format='pdf',dpi=120,bbox_inches='tight')

#arnoldi(f,Uk,U0,100,eig_estimate)
