import numpy as np

import pickle
import time


def sinkhorn(a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, beta=1.):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    dim_a = len(a)
    dim_b = len(b)  

    if log:
        log = {'err': []}#, 'loss': []}
    
    u = np.zeros(dim_a, dtype=np.float64) #/ dim_a; #u = np.log(u) # log to match linear-scale sinkhorn initializations
    v = np.zeros(dim_b, dtype=np.float64) #/ dim_b; #v = np.log(v)


    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    #B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
    ones_a = np.ones(dim_a, dtype=np.float64)
    ones_b = np.ones(dim_b, dtype=np.float64)
    
    cpt = 0
    err = 1
    
    def f(xk):        
        u,v = xk[0:dim_a,0],xk[dim_a:xk.size,0]
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))                
        
        uprev = u
        vprev = v        
        u = uprev + np.log(a) - np.log(np.matmul(ones_b, B.T)+ 1e-299) 
        v = vprev
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
        
        #err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
        #            np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
        #print('* {:8e}|'.format(err)) 
        
        uprev = u
        vprev = v                
        v = vprev + np.log(b) - np.log(np.matmul(ones_a, B)+ 1e-299)
        u = uprev        
        rk = np.concatenate((u,v)).reshape(xk.shape)-xk
        return np.linalg.norm(rk),rk.reshape(rk.size,1),B
                
    
    cnt = 0
    N = dim_a+dim_b
    xk = np.concatenate((u,v)).reshape(N,1)
    fk,res,B = f(xk)
            
    res_v = []               
    err_v = []
    time_v = []
    start = time.time()
    
    while (err > stopThr and cpt < numItermax):
        
        #print(cpt,fk)
        res_v.append(fk)        
        time_v.append(time.time()-start)
        err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
                    np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
        err_v.append(err)
                                                                                     
        xk += beta*res        
        

        if (np.any(np.isnan(u)) or np.any(np.isinf(u))
                or np.any(np.isnan(v)) or np.any(np.isinf(v))
                or np.any(np.isnan(B)) or np.any(np.isinf(B))):            
            print('Warning: numerical errors at iteration', cpt)
            uprev,vprev = x_prev[0:dim_a,0],x_prev[dim_a:N,0]
            u = uprev
            v = vprev
            B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
            break 
            
        if cpt % 10 == 0: #10            
            err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
                    np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 2
        
        cnt = cnt+1
        fk,res,B = f(xk)
        
    '''err_tmp = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
            np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
    err_v.append(err_tmp)
    res_v.append(fk) '''
    
    if log:
        log['u'] = u
        log['v'] = v        
    
    pickle.dump((res_v,err_v,time_v),open('sinkhorn.ser','wb'))
    
    if log:
        return B, cpt, log
    else:
        return B, cpt


def anderson2_sinkhorn(a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, beta=1.,m=20,type2=True):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    dim_a = len(a)
    dim_b = len(b)  

    if log:
        log = {'err': []}#, 'loss': []}
    
    u = np.zeros(dim_a, dtype=np.float64) #/ dim_a; #u = np.log(u) # log to match linear-scale sinkhorn initializations
    v = np.zeros(dim_b, dtype=np.float64) #/ dim_b; #v = np.log(v)


    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    #B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
    ones_a = np.ones(dim_a, dtype=np.float64)
    ones_b = np.ones(dim_b, dtype=np.float64)
    
    cpt = 0
    err = 1
    
    def f(xk):        
        u,v = xk[0:dim_a,0],xk[dim_a:xk.size,0]
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))                
        
        uprev = u
        vprev = v        
        u = uprev + np.log(a) - np.log(np.matmul(ones_b, B.T)+ 1e-299) 
        v = vprev
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
        
        #err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
        #            np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
        #print('* {:8e}|'.format(err)) 
        
        uprev = u
        vprev = v                
        v = vprev + np.log(b) - np.log(np.matmul(ones_a, B)+ 1e-299)
        u = uprev        
        rk = np.concatenate((u,v)).reshape(xk.shape)-xk
        return np.linalg.norm(rk),rk.reshape(rk.size,1),B
                
    
    cnt = 0
    N = dim_a+dim_b
    xk = np.concatenate((u,v)).reshape(N,1)
    fk,res,B = f(xk)

    Xk = np.zeros((N,m))
    Rk = np.zeros((N,m))
    Zk = Rk if type2 else Xk
    mk = 0
    res_v = []               
    err_v = []
    time_v = []
    start = time.time()
    
    while (err > stopThr and cpt < numItermax):
        
        #print(cpt,fk)
        res_v.append(fk)        
        time_v.append(time.time()-start)
        err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
            np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
        err_v.append(err)

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
        
        if cnt > 0:
            Xk_tmp, Rk_tmp, Zk_tmp = Xk[:,0:mk],Rk[:,0:mk],Zk[:,0:mk]
            Gamma = np.linalg.solve(Zk_tmp.T@Rk_tmp,Zk_tmp.T@res)
            xk_bar = xk-Xk_tmp @ Gamma
            rk_bar = res-Rk_tmp @ Gamma
            xk = xk_bar + beta*rk_bar 
        else:                                                        
            xk += beta*res        
        

        if (np.any(np.isnan(u)) or np.any(np.isinf(u))
                or np.any(np.isnan(v)) or np.any(np.isinf(v))
                or np.any(np.isnan(B)) or np.any(np.isinf(B))):            
            print('Warning: numerical errors at iteration', cpt)
            uprev,vprev = x_prev[0:dim_a,0],x_prev[dim_a:N,0]
            u = uprev
            v = vprev
            B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
            break 
            
        if cpt % 10 == 0: #10            
            err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
                    np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 2
        
        cnt = cnt+1
        fk,res,B = f(xk)

    '''err_tmp = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
            np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
    err_v.append(err_tmp)
    res_v.append(fk) '''
        
    if log:
        log['u'] = u
        log['v'] = v
    
    filename = 'anderson2_' if type2 else 'anderson1_'
    pickle.dump((res_v,err_v,time_v),open(filename+str(m)+'.ser','wb'))
    
    if log:
        return B, cpt, log
    else:
        return B, cpt
        

def matvec(f,x_star,v):
    epsilon = 1e-8
    #epsilon /= np.linalg.norm(v,ord=2)
    #epsilon *= np.linalg.norm(x_star,ord=2)    
    fk,rk,_ = f(x_star)
    x_next = x_star+epsilon*v
    fk2,rk2,_ = f(x_next)
    Av = -(rk2 - rk) / epsilon
    return np.linalg.norm(Av,ord=2), Av

def arnoldi(f, x_star,x0, maxit=100):
    cnt = 0
    xk = x0.copy()
    fk, rk,_ = f(xk)

    beta = np.linalg.norm(rk,ord=2)
    Vk = [rk / beta]
    H = []
    flag = False
    while cnt < maxit and (not flag):
        v = Vk[-1]
        #w = A @ v
        _, w = matvec(f,x_star,v)
        hs = []
        for v in Vk:
            h = np.vdot(w.reshape(-1), v.reshape(-1))
            w -= h * v
            hs.append(h)
        h = np.linalg.norm(w.reshape(-1),ord=2)
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

    eig = np.linalg.eigvals(Hk)
    

    print("Arnoldi method for eigenvalue computation.")
    return eig

        
def am2_sinkhorn(a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, tao=1e-16, beta=1.,m=10,type2=True):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    dim_a = len(a)
    dim_b = len(b)  

    if log:
        log = {'err': []}#, 'loss': []}
    
    u = np.zeros(dim_a, dtype=np.float64) #/ dim_a; #u = np.log(u) # log to match linear-scale sinkhorn initializations
    v = np.zeros(dim_b, dtype=np.float64) #/ dim_b; #v = np.log(v)


    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    #B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
    ones_a = np.ones(dim_a, dtype=np.float64)
    ones_b = np.ones(dim_b, dtype=np.float64)        
    
    cpt = 0
    err = 1
    
    def f(xk):        
        u,v = xk[0:dim_a,0],xk[dim_a:xk.size,0]
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))                
        
        uprev = u
        vprev = v        
        u = uprev + np.log(a) - np.log(np.matmul(ones_b, B.T)+ 1e-299) 
        v = vprev
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))                
        
        uprev = u
        vprev = v                
        v = vprev + np.log(b) - np.log(np.matmul(ones_a, B)+ 1e-299)
        u = uprev        
        rk = np.concatenate((u,v)).reshape(xk.shape)-xk
        return np.linalg.norm(rk),rk.reshape(rk.size,1),B
                
    
    cnt = 0
    N = dim_a+dim_b
    xk = np.concatenate((u,v)).reshape(N,1)
    x0 = xk.copy()    
    fk,rk,B = f(xk)

    N = xk.size
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
    err_v = []
    time_v = []
    start = time.time()
    
    restart_cnt = 0           
    
    while (err > stopThr and cpt < numItermax):
        
        #print(cpt,fk)
        res_v.append(fk) 
        time_v.append(time.time()-start)
        err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
            np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
        err_v.append(err) 

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

            qq = np.vdot(q, q)
            qqs.append(qq)
            if qqs[-1] <= (tao) * qqs[0] or mk == m:
                restart = True
                mk = -1
                qqs = []
                beta_prev = beta
                restart_cnt = restart_cnt+1
                #print('restart!',cpt,cnt)

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

                eig = np.linalg.eigvals(Hk)
                beta_prev = beta
                #beta = 1. / abs(max(eig))
                #beta = 2. / max(abs(eig))
                #print('rho:',max(abs(eig)))


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
        

        if (np.any(np.isnan(u)) or np.any(np.isinf(u))
                or np.any(np.isnan(v)) or np.any(np.isinf(v))
                or np.any(np.isnan(B)) or np.any(np.isinf(B))):            
            print('Warning: numerical errors at iteration', cpt)
            uprev,vprev = x_prev[0:dim_a,0],x_prev[dim_a:N,0]
            u = uprev
            v = vprev
            B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
            break 
            
        if cpt % 10 == 0: #10            
            err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
                    np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 2
        
        cnt = cnt+1
        fk,rk,B = f(xk)
        
    print('restart cnt:',restart_cnt)
    print('|xk-x0|_2:',np.linalg.norm(xk-x0))
    #res_v.append(fk)    
    '''if eig.size>0:
        print(np.sort(abs(eig)))
        print('rho(F):',max(abs(eig)))

    eig_tmp = arnoldi(f,xk.copy(),x0.copy(),mk)    
    if eig_tmp.size > 0:
        #print(np.sort(abs(eig_tmp)))
        print('*rho(F):', max(abs(eig_tmp)))

    import matplotlib.pyplot as plt
    print(len(eig_tmp), len(eig))
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='red', marker='o')
    plt.scatter(eig.real, eig.imag, label='AM Ritz value', color='blue', marker='+')
    plt.legend()
    plt.show()
    #pickle.dump((eig,eig_tmp), open('am2_eig_' + str(omega) + '.ser', 'wb')) '''

    filename = 'am2_' if type2 else 'am1_'
    pickle.dump((res_v,err_v,time_v),open(filename+str(m)+'.ser','wb'))        
    if log:
        log['u'] = u
        log['v'] = v
    
    if log:
        return B, cpt, log
    else:
        return B, cpt   


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


def iram_sinkhorn(a, b, M, reg, numItermax=1000,
                   stopThr=1e-9, verbose=False, log=False, tao=1e-16, beta=1.,m=20,n=10,mit=100,type2=True):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    dim_a = len(a)
    dim_b = len(b)  

    if log:
        log = {'err': []}#, 'loss': []}
    
    u = np.zeros(dim_a, dtype=np.float64) #/ dim_a; #u = np.log(u) # log to match linear-scale sinkhorn initializations
    v = np.zeros(dim_b, dtype=np.float64) #/ dim_b; #v = np.log(v)


    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty(M.shape, dtype=M.dtype)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    #B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
    ones_a = np.ones(dim_a, dtype=np.float64)
    ones_b = np.ones(dim_b, dtype=np.float64)        
    
    cpt = 0
    err = 1
    
    def f(xk):        
        u,v = xk[0:dim_a,0],xk[dim_a:xk.size,0]
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))                
        
        uprev = u
        vprev = v        
        u = uprev + np.log(a) - np.log(np.matmul(ones_b, B.T)+ 1e-299) 
        v = vprev
        B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))                
        
        uprev = u
        vprev = v                
        v = vprev + np.log(b) - np.log(np.matmul(ones_a, B)+ 1e-299)
        u = uprev        
        rk = np.concatenate((u,v)).reshape(xk.shape)-xk
        return np.linalg.norm(rk),rk.reshape(rk.size,1),B
                
    
    cnt = 0
    N = dim_a+dim_b
    xk = np.concatenate((u,v)).reshape(N,1)
    x0 = xk.copy()    
    fk,rk,B = f(xk)

    N = xk.size
    Pk = np.zeros((N, m+1))
    Qk = np.zeros((N, m+1))
    Vk = Qk if type2 else Pk

    Hk_bar = np.zeros((m + 1, m))    
    mk = -1
    Mk = -1

    nd = fk    
    res_v = []
    err_v = []
    min2_v = []
    min3_v = []
    time_v = []
    start = time.time()
    
    restart = False
    vqs = []

    beta_prev = beta
    restart_cnt = 0  
    
    min_tmp2 = 1e10
    min_tmp3 = 1e10
    
    while (err > stopThr and cpt < numItermax):
        
        #print('cnt:',cnt,fk)
        res_v.append(fk)
        time_v.append(time.time()-start)
        err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
            np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)
        err_v.append(err)

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
            
            min_tmp2 = min(min_tmp2,vqs[-1]/vqs[0])
            min2_v.append(min_tmp2)
            
            if vqs[-1]/vqs[0] <= tao or Mk == mit:
                restart = True
                mk = -1                        
                Mk = -1                  

            if mk >= 1:
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
                
                min_tmp3 = min(min_tmp3,abs(1-gammak[mk-1]))
                min3_v.append(min_tmp3)

                Hk = Hk_bar[0:mk, 0:mk]

                eig = np.linalg.eigvals(Hk)                                                 
                beta_prev = beta
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
                                            
                    VQ = Vk[:,0:mk+1].T@Qk[:,0:mk+1]                                                      
                    
        if restart:            
            mk = -1
            Mk = -1
            vqs = []
            beta_prev = beta
            restart_cnt = restart_cnt+1
            #print('restart!')         
        

        x_prev = xk.copy()
        r_prev = rk.copy()        

        mk = mk+1
        Mk = Mk+1
        
        if mk>0:            
            P_tmp = Pk[:,0:mk]
            Q_tmp = Qk[:,0:mk]
            V_tmp = Vk[:,0:mk]
            gammak = np.linalg.solve(VQ,V_tmp.T@rk) 
            xk = xk-P_tmp@gammak
            rk = rk-Q_tmp@gammak                                          

        xk += beta * rk  # rk_bar = rk                      
                                        

        if (np.any(np.isnan(u)) or np.any(np.isinf(u))
                or np.any(np.isnan(v)) or np.any(np.isinf(v))
                or np.any(np.isnan(B)) or np.any(np.isinf(B))):            
            print('Warning: numerical errors at iteration', cpt)
            uprev,vprev = x_prev[0:dim_a,0],x_prev[dim_a:N,0]
            u = uprev
            v = vprev
            B = np.einsum('i,ij,j->ij', np.exp(u), K, np.exp(v))
            break 
            
        if cpt % 10 == 0: #10            
            err = np.linalg.norm(np.matmul(ones_b, B.T)-a, ord=1) + \
                    np.linalg.norm(np.matmul(ones_a, B)-b, ord=1)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 2
        
        cnt = cnt+1
        fk,rk,B = f(xk)
        
    print('restart cnt:',restart_cnt)
    print('|xk-x0|_2:',np.linalg.norm(xk-x0))
    print('min(v):',min_tmp2)
    print('min(1-Gammak[k]):',min_tmp3)
    #res_v.append(fk)    
    '''if eig.size>0:
        print(np.sort(abs(eig)))
        print('rho(F):',max(abs(eig)))

    eig_tmp = arnoldi(f,xk.copy(),x0.copy(),mk)    
    if eig_tmp.size > 0:
        #print(np.sort(abs(eig_tmp)))
        print('*rho(F):', max(abs(eig_tmp)))

    import matplotlib.pyplot as plt
    print(len(eig_tmp), len(eig))
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='red', marker='o')
    plt.scatter(eig.real, eig.imag, label='AM Ritz value', color='blue', marker='+')
    plt.legend()
    plt.show()
    #pickle.dump((eig,eig_tmp), open('am2_eig_' + str(omega) + '.ser', 'wb')) '''
        
    if log:
        log['u'] = u
        log['v'] = v
    
    if type2:
        filename = 'iram2_'
    else:
        filename = 'iram1_'
    print(len(res_v),len(err_v))
    pickle.dump((res_v,err_v,time_v), open(filename+str(m)+'_'+str(n)+'_'+str(mit)+'.ser', 'wb'))
    pickle.dump((min2_v,min3_v), open(filename+'min_'+str(m)+'_'+str(n)+'_'+str(mit)+'.ser', 'wb'))
    
    if log:
        return B, cpt, log
    else:
        return B, cpt 
