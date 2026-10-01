import numpy as np

from scipy.linalg import eigvalsh_tridiagonal, eigvalsh

# np.random.seed(1)
# D_in, D_out = 100, 100
#
# A = np.random.randn(D_in,D_out)
# A = A.T@A
# x = np.random.randn(D_out,1)
# d = A @ x

#import pickle
#with open('data2.ser','wb') as f:
#    pickle.dump((A,x,d),f)

#import pickle
#with open('data2.ser','rb') as f:
#    A,x,d = pickle.load(f)

#D_in,D_out = A.shape

np.random.seed(1)

D_in, D_out = 1000,1000

'''A = np.random.randn(D_in,D_out)
D,V = np.linalg.eig(A)
D.real = abs(D.real)
A = V@np.diag(D)@np.linalg.inv(V)
A = A.real 
x = np.random.randn(D_out,1)
d = A @ x '''

d = np.zeros(D_out)
d[0],d[1] = 0.01,0.1
#d[0],d[1] = -0.01,-0.1
#d[0],d[1] = -100,-10
#d[0],d[1] = -200,-100
d[2:D_out] = np.linspace(1,D_out-2,D_out-2)
d2 = np.ones(D_out-1)
#Q,_ = np.linalg.qr(np.random.randn(D_in,D_out))
A = np.diag(d)+np.diag(d2,1)
#A = Q.T@A@Q
d = np.ones((D_out,1))

# np.random.seed(1)

'''D_in, D_out = 100, 100
A1 = np.random.randn(D_in,D_out)
Q1,_ = np.linalg.qr(A1)
A2 = np.random.randn(D_in,D_out)
Q2,_ = np.linalg.qr(A2)
D = np.linspace(1,100,100)
D = np.diag(D)
A = Q1.T @ (D @ Q1)
x = np.random.randn(D_out,1)
d = A @ x '''
import pickle
D_in,D_out = A.shape

def linear(x):
    return A @ x

def misfit(x):
    u = linear(x)
    return 0.5*(u-d).T.dot(u-d)

def linear_gradient(x):
    return A.T @ (A@x-d)

def misfit_gradient(x):
    return linear_gradient(x)

def norm2(x):
    return np.linalg.norm(x,ord=2)

def misfit_total(x):
    gk = A@x-d
    return norm2(gk),gk

    
from numpy.linalg import pinv


def picard(fh, x0, tol=1e-8, maxit=1e5, beta=0.001):
    xk = x0
    fk, gk = fh(xk)
    r = -gk
    nd = norm2(r)
    relres = tol + 1
    relres_v = []

    cnt = 0
    while cnt < maxit and relres > tol:
        r = -gk
        relres = norm2(r) / nd
        xk = xk + beta * r

        relres = norm2(r) / nd
        if cnt > 0:
            relres_v.append(relres)
        cnt = cnt + 1
        fk, gk = fh(xk)
        print(cnt, "%e %e" % (norm2(gk) / nd, fk))
    fk, gk = fh(xk)
    #relres = norm2(gk) / nd
    #relres_v.append(relres)
    #print(relres_v)
    with open('picard.ser', 'wb') as f:
        pickle.dump(relres_v, f)

    print('picard method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (relres, fk, cnt))

def anderson2(fh,x0,tol=1e-8,maxit=1e5,beta=1.0,m=100):
    cnt = 0
    xk = x0
    fk,gk = fh(xk)

    N = len(x0)
    Xk = np.zeros((N,m))
    Rk = np.zeros((N,m))

    nd = norm2(gk)
    relres = tol+1
    relres_v = []
    relres_bar_v = []
    #misf_v = []

    while cnt < maxit and relres>tol:
        res = -gk
        #print(norm2(res))
        relres = norm2(res)/nd
        if cnt >0:
            relres_v.append(relres)
            #misf_v.append(fk)

        if cnt >= 1:
            k = (cnt-1) % m
            Xk[:,k] = (xk-x_prev).reshape(len(xk))
            Rk[:,k] = (res-res_prev).reshape(len(xk))
        x_prev = xk.copy()
        res_prev = res.copy()

        if cnt == 0:
            xk += beta*res
            #print('gd: ')
        else:            
            #Gamma = pinv(Rk.T@Rk)@(Rk.T@res)            
            Gamma = np.linalg.solve(Rk[:,0:cnt].T@Rk[:,0:cnt],Rk[:,0:cnt].T@res)
            xk_bar = xk-Xk[:,0:cnt] @ Gamma
            rk_bar = res-Rk[:,0:cnt] @ Gamma
            xk = xk_bar + beta*rk_bar            
            relres_bar_v.append(norm2(rk_bar)/nd)
        cnt = cnt+1
        fk,gk = fh(xk)
        if cnt == 1:
            rk_bar = res
        if cnt>1:
            print(cnt-1,"%e %e %e" %(fk,norm2(gk)/nd, norm2(rk_bar)/nd))
    #print(relres_v)
    fk,gk = fh(xk)
    #relres = norm2(gk) / nd
    #relres_v.append(relres)
    with open('anderson2_'+str(m)+'.ser','wb') as f:
        assert len(relres_v) == len(relres_bar_v)
        pickle.dump(relres_v,f)
    
    print('Anderson2 method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (relres,fk,cnt))
    return relres_v


def anderson1(fh, x0, tol=1e-8, maxit=1e5, beta=1.0, m=100):
    cnt = 0
    xk = x0
    fk, gk = fh(xk)

    N = len(x0)
    Xk = np.zeros((N, m))
    Rk = np.zeros((N, m))

    nd = norm2(gk)
    relres = tol + 1
    relres_v = []
    relres_bar_v = []
    #misf_v = []

    while cnt < maxit and relres > tol:
        res = -gk
        # print(norm2(res))
        relres = norm2(res) / nd
        if cnt > 0:
            relres_v.append(relres)
            #misf_v.append(fk)

        if cnt >= 1:
            k = (cnt - 1) % m
            Xk[:, k] = (xk - x_prev).reshape(len(xk))
            Rk[:, k] = (res - res_prev).reshape(len(xk))
        x_prev = xk.copy()
        res_prev = res.copy()

        if cnt == 0:
            xk += beta * res
            # print('gd: ')
        else:
            Gamma = np.linalg.solve(Xk[:,0:cnt].T@Rk[:,0:cnt],Xk[:,0:cnt].T@res)
            xk_bar = xk-Xk[:,0:cnt] @ Gamma
            rk_bar = res-Rk[:,0:cnt] @ Gamma

			#Gamma = pinv(Xk.T @ Rk) @ (Xk.T @ res)
			#xk_bar = xk - Xk @ Gamma
            #rk_bar = res - Rk @ Gamma
            xk = xk_bar + beta * rk_bar
            relres_bar_v.append(norm2(rk_bar)/nd)
        cnt = cnt + 1
        fk, gk = fh(xk)
        if cnt == 1:
            rk_bar = res
        if cnt > 1:
            print(cnt - 1, "%e %e %e" % (fk, norm2(gk) / nd, norm2(rk_bar) / nd))
    # print(relres_v)
    fk, gk = fh(xk)
    #relres = norm2(gk) / nd
    #relres_v.append(relres)
    with open('anderson1_'+str(m)+'.ser','wb') as f:
        assert len(relres_v) == len(relres_bar_v)
        pickle.dump(relres_v,f)

    print('Anderson1 method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (
    relres, fk, cnt))
    return relres_v


def am2(fh, x0, tol=1e-8, maxit=1e5, tao=1e-16, beta=1.0, m=100):
    cnt = 0
    xk = x0
    fk, gk = fh(xk)

    N = len(x0)
    Pk = np.zeros((N, m))
    Qk = np.zeros((N, m))

    Hk_bar = np.zeros((m + 1, m))
    zetas = np.zeros(m)
    gammas = np.zeros(m)
    xis = np.zeros(m)
    #cur = -1
    #length = 0
    mk = -1

    nd = norm2(gk)
    relres = tol + 1
    relres_v = []
    relres_bar_v = []
    #misf_v = []
    # restart = False
    qqs = []

    beta_prev = beta
    while cnt < maxit and relres > tol:
        rk = -gk
        # print(norm2(res))
        relres = norm2(rk) / nd
        if cnt > 0:
            relres_v.append(relres)
            #misf_v.append(fk)

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
            #print('#',qq)
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
                #beta = 2./ max(abs(eig))

                Q_tmp = Qk[:, 0:mk]
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
                # print(np.linalg.norm(Hk),np.linalg.norm(H_tmp))


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

        if cnt>0:
            relres_bar_v.append(norm2(rk) / nd)

        xk += beta * rk  # rk_bar = rk
        cnt = cnt + 1

        fk, gk = fh(xk)

        if cnt > 1:
            print(cnt - 1, "%e %e %e" % (fk, norm2(gk) / nd, norm2(rk) / nd))
            # print(relres_v)
    fk, gk = fh(xk)
    #relres = norm2(gk) / nd
    #relres_v.append(relres)
    with open('am2_'+str(m)+'.ser', 'wb') as f:
        assert len(relres_v) == len(relres_bar_v)
        pickle.dump(relres_v, f)

    '''eig_tmp = np.linalg.eigvals(A)
    print('****************')
    # print(eig_tmp)

    import matplotlib.pyplot as plt
    print(len(eig_tmp), len(eig))
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='red', marker='o')
    plt.scatter(eig.real, eig.imag, label='AM2', color='blue', marker='+')
    plt.legend()
    plt.show() '''
    #plt.savefig('figure_am2_linear.pdf',format='pdf',dpi=120,bbox_inches='tight') 
    print('am2 method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (
            norm2(gk) / nd, fk, cnt))    


def am1(fh, x0, tol=1e-8, maxit=1e5, tao=1e-16, beta=1.0, m=100):
    cnt = 0
    xk = x0
    fk, gk = fh(xk)

    N = len(x0)
    Pk = np.zeros((N, m))
    Qk = np.zeros((N, m))

    Hk_bar = np.zeros((m + 1, m))
    zetas = np.zeros(m)
    gammas = np.zeros(m)
    xis = np.zeros(m)
    #cur = -1
    #length = 0
    mk = -1

    nd = norm2(gk)
    relres = tol + 1
    relres_v = []
    relres_bar_v = []
    #misf_v = []
    # restart = False
    pqs = []

    beta_prev = beta
    while cnt < maxit and relres > tol:
        rk = -gk
        # print(norm2(res))
        relres = norm2(rk) / nd
        if cnt > 0:
            relres_v.append(relres)
            #misf_v.append(fk)

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
            #print('#',pq)
            if pqs[-1] <= (tao) * pqs[0] or mk == m:
                restart = True
                mk = -1
                pqs = []
                beta_prev = beta
                print('restart!')

            # if mk>=1 and abs(1-gammas[mk-1])<1e-10:
            #     restart = True
            #     mk = -1
            #     print('restart2!')

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
                #beta = 2 / max(abs(eig))

                Q_tmp = Qk[:, 0:mk]
                P_tmp = Pk[:, 0:mk]

                H_tmp = np.linalg.inv(P_tmp.T @ Q_tmp) @ (P_tmp.T @ (A @ Q_tmp))
                # for j in range(0,H_tmp.shape[1]):
                #    H_tmp[j+2:,j].fill(0)
                # print(H_tmp[-1,:])
                #print('*', np.linalg.norm(Hk - H_tmp) / np.linalg.norm(H_tmp))
                # eig = np.linalg.eigvals(H_tmp)
                # beta_prev = beta
                # beta = 1./abs(max(eig))
                # print(eig)
                # print(H_tmp)
                # print(np.linalg.norm(Hk),np.linalg.norm(H_tmp))


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

        if cnt > 0:
            relres_bar_v.append(norm2(rk) / nd)

        xk += beta * rk  # rk_bar = rk
        cnt = cnt + 1

        fk, gk = fh(xk)

        if cnt > 1:
            print(cnt - 1, "%e %e %e" % (fk, norm2(gk) / nd, norm2(rk) / nd))
            # print(relres_v)
    fk, gk = fh(xk)
    #relres = norm2(gk) / nd
    #relres_v.append(relres)
    with open('am1_'+str(m)+'.ser', 'wb') as f:
        assert len(relres_v) == len(relres_bar_v)
        pickle.dump(relres_v, f)

    '''eig_tmp = np.linalg.eigvals(A)
    print('****************')
    # print(eig_tmp)
    #print(eig)
    #print(eig_tmp)
    import matplotlib.pyplot as plt
    print(len(eig_tmp), len(eig))
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='red', marker='o')
    plt.scatter(eig.real, eig.imag, label='AM1', color='blue', marker='+')
    plt.legend()
    plt.show() '''
    #plt.savefig('figure_am1_linear.pdf',format='pdf',dpi=120,bbox_inches='tight') 
    print('am1 method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (
            norm2(gk) / nd, fk, cnt))
    

def givens(a,b):
    if b == 0:
        c,s = 1,0
    else:
        if abs(b)>abs(a):
            t = a/b
            s = 1/((1+t**2)**0.5)
            c = s*t
        else:
            t = b/a
            c = 1/((1+t**2)**0.5)
            s = c*t
    return c,s


def gmres(fh,x0,tol=1e-8,maxit=1e5):
    xk = x0
    fk,gk = fh(xk)
    rk = -gk
    nd = norm2(rk)
    relres = tol+1
    relres_v = []
    
    cnt =0 
    
    beta = norm2(rk)
    Vk = [rk/beta]
    Hk = []    
    rhs = [beta]
    rotates = []
    flag = False
    while cnt < maxit and relres>tol and (not flag):
        v = Vk[-1]
        w = A@v
        hs = []
        for v in Vk:
            h = np.vdot(w,v)
            w -= h*v
            hs.append(h)
        h = norm2(w)
        hs.append(h)
        Hk.append(hs)
        if h == 0:
            m = cnt+1           
            flag = True
        else:
            Vk.append(w/h)
        rhs.append(0)                
                
        for j in range(0,cnt):
            (c,s) = rotates[j]
            t1,t2 = hs[j],hs[j+1]
            hs[j] = c*t1+s*t2
            hs[j+1] = -s*t1+c*t2                    
        t1,t2 = hs[-2],hs[-1]
        c,s = givens(t1,t2)        
        hs[-2] = c*t1+s*t2
        hs[-1] = -s*t1+c*t2
        rotates.append((c,s))        
        
        t1,t2 = rhs[cnt],rhs[cnt+1]
        rhs[cnt] = c*t1+s*t2
        rhs[cnt+1] = -s*t1+c*t2
        nr = abs(rhs[-1])
        relres = nr/nd
        relres_v.append(relres)
        
        #print(Hk)
        y = rhs[0:-1].copy()
        for j in range(cnt,-1,-1):
            y[j] /= Hk[j][j]
            for k in range(0,j):
                y[k] -= Hk[j][k]*y[j]
        
        xk = x0.copy()
        for j in range(0,len(Vk)-1):
            xk += Vk[j]*y[j]
        
        cnt = cnt+1
        fk,gk = fh(xk)
        print(cnt,"%e %e %e" %(fk,norm2(gk)/nd,relres))
    
    #relres = norm2(gk) / nd
    #relres_v.append(relres)
    with open('gmres_unsym.ser','wb') as f:
        pickle.dump(relres_v,f)

    print('GMRES method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (relres,fk,cnt))   


def arnoldi(fh,x0,tol=1e-8,maxit=1e5):
    xk = x0
    fk,gk = fh(xk)
    rk = -gk
    nd = norm2(rk)
    relres = tol+1
    relres_v = []
    
    cnt =0 
    
    beta = norm2(rk)
    Vk = [rk/beta]
    Hk = []    
    rhs = [beta]
    rotates = []
    flag = False
    while cnt < maxit and relres>tol and (not flag):
        v = Vk[-1]
        w = A@v
        hs = []
        for v in Vk:
            h = np.vdot(w,v)
            w -= h*v
            hs.append(h)
        h = norm2(w)
        hs.append(h)
        Hk.append(hs)
        if h == 0:
            m = cnt+1           
            flag = True
        else:
            Vk.append(w/h)
        rhs.append(0)                
                
        for j in range(0,cnt):
            (c,s) = rotates[j]
            t1,t2 = hs[j],hs[j+1]
            hs[j] = c*t1+s*t2
            hs[j+1] = -s*t1+c*t2                            
        
        #print(Hk)
        y = rhs[0:-1].copy()
        for j in range(cnt,-1,-1):
            y[j] /= Hk[j][j]
            for k in range(0,j):
                y[k] -= Hk[j][k]*y[j]
        
        nr = abs(hs[-1]*y[-1])
        relres = nr/nd
        relres_v.append(relres)
        
        xk = x0.copy()
        for j in range(0,len(Vk)-1):
            xk += Vk[j]*y[j]
            
        t1,t2 = hs[-2],hs[-1]
        c,s = givens(t1,t2)        
        hs[-2] = c*t1+s*t2
        hs[-1] = -s*t1+c*t2
        rotates.append((c,s))        
        
        t1,t2 = rhs[cnt],rhs[cnt+1]
        rhs[cnt] = c*t1+s*t2
        rhs[cnt+1] = -s*t1+c*t2        
        
        cnt = cnt+1
        fk,gk = fh(xk)
        print(cnt,"%e %e %e" %(fk,norm2(gk)/nd,relres))

    #relres = norm2(gk) / nd
    #relres_v.append(relres)
    with open('arnoldi_unsym.ser', 'wb') as f:
        pickle.dump(relres_v, f)
    print('Arnoldi method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (relres,fk,cnt))


def lu_crout(A):    
    n = A.shape[0]
    A = A.copy()
    for k in range(0,n-1):
        A[k,k+1:n] = A[k,k+1:n]/A[k,k]
        A[k+1:n,k+1:n] = A[k+1:n,k+1:n] - A[k+1:n,k:k+1]@A[k:k+1,k+1:n]   #            
    return A
    
def lu_solve(A,b):
    n = A.shape[0]
    b[0] = b[0]/A[0,0]
    for i in range(1,n):
        b[i] = (b[i]-np.dot(A[i,0:i],b[0:i]))/A[i,i]
    for i in range(n-2,-1,-1):
        b[i] = b[i]-np.dot(A[i,i+1:n],b[i+1:n])
    return b

def qr_single3(H,a):
    n = H.shape[0]  
    H0 = H.copy()  
    H[np.diag_indices(n)] -= a    
    rotates = []
    for k in range(0,n-1):
        c,s = givens(H[k,k],H[k+1,k])        
        rotate = np.array([[c,s],[-s,c]])
        H[k:k+2,k:n] = rotate@H[k:k+2,k:n]
        rotates.append(rotate)
    print('1:',H)
    for k in range(0,n-1):
        H[0:k+2,k:k+2] = H[0:k+2,k:k+2]@rotates[k].T
    H[np.diag_indices(n)] += a
    print('2:',H)
    Q = np.eye(n)
    for k in range(0,n-1):
        tmp = np.eye(n)
        tmp[k:k+2,k:k+2] = rotates[k].T
        Q = Q@tmp
    print(np.allclose(H,Q.T@H0@Q)) 

def qr_single2(H,a):
    n = H.shape[0]  
    H0 = H.copy()  
    x = H[0,0]-a    
    y = H[1,0]    
    rotates = []
    for k in range(0,n-2):
        c,s = givens(x,y)        
        rotate = np.array([[c,s],[-s,c]])
        q = max(k-1,0)
        H[k:k+2,q:n] = rotate@H[k:k+2,q:n]
        H[0:k+3,k:k+2] = H[0:k+3,k:k+2]@rotate.T
        rotates.append(rotate)
        x = H[k+1,k]
        if k<n-2:
            y = H[k+2,k]
    c,s = givens(x,y)        
    rotate = np.array([[c,s],[-s,c]])
    H[n-2:n,n-3:n] = rotate@H[n-2:n,n-3:n]
    H[0:n,n-2:n] = H[0:n,n-2:n]@rotate.T
    rotates.append(rotate)
    print('H1:',H)      
    Q = np.eye(n)
    for k in range(0,n-1):
        tmp = np.eye(n)
        tmp[k:k+2,k:k+2] = rotates[k].T
        Q = Q@tmp
    H2 = Q.T@H0@Q
    print(np.allclose(H2,H)) 


def house(x):
    m = len(x)
    sigma = np.vdot(x[1:m],x[1:m])
    v = x.copy()
    v[0] = 1
    if sigma == 0 and x[0] >= 0:
        beta = 0
    elif sigma == 0 and x[0] < 0:
        beta = 2
    else:
        mu = (x[0]**2+sigma)**0.5
        v[0] = x[0]-mu if x[0]<=0 else -sigma/(x[0]+mu)
        beta = 2*v[0]**2/(sigma+v[0]**2)
        v /= v[0]
    return v,beta               

'''def qr_double(H,a1,a2):
    assert(np.conj(a1) == a2)
    n = H.shape[0]
    m = n-2
    s = a1+a2
    t = np.abs(a1)**2
    x = H[0,0]*H[0,0]+H[0,1]*H[1,0]-s*H[0,0]+t
    y = H[1,0]*(H[0,0]+H[1,1]-s)
    z = H[1,0]*H[2,1]
    for k in range(-1,n-3):
        v,beta = house(np.array([[x],[y],[z]]))
        q = max(0,k)
        H[k+1:k+4,q:n] = H[k+1:k+4,q:n]-beta*v@(v.T@H[k+1:k+4,q:n])
        r = min(k+5,n)
        H[0:r,k+1:k+4] = H[0:r,k+1:k+4]-beta*(H[0:r,k+1:k+4]@v)@v.T
        x = H[k+2,k+1]
        y = H[k+3,k+1]
        if k<n-4:
            z = H[k+4,k+1]
    v,beta = house(np.array([[x],[y]]))
    H[n-2:n,n-3:n] = H[n-2:n,n-3:n]-beta*v@(v.T@H[n-2:n,n-3:n])
    H[0:n,n-2:n] = H[0:n,n-2:n]-beta*(H[0:n,n-2:n]@v)@v.T '''
    
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

    

def iram(fh, x0, tol=1e-8, maxit=1e5, tao=1e-16, beta=1.0, m=10,n=2,mit=100, type2=True):
    cnt = 0
    xk = x0
    fk, gk = fh(xk)

    N = len(x0)
    Pk = np.zeros((N, m+1))
    Qk = np.zeros((N, m+1))
    Vk = Qk if type2 else Pk

    Hk_bar = np.zeros((m + 1, m))    
    mk = -1

    nd = norm2(gk)
    relres = tol + 1
    relres_v = []
    relres_bar_v = []    
    # restart = False
    vqs = []

    beta_prev = beta
    while cnt < maxit and relres > tol:
        rk = -gk
        relres = norm2(rk) / nd
        if cnt > 0:
            relres_v.append(relres)

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
            vq = VQ[-1,-1]-VQ[mk:mk+1,0:mk]@np.linalg.solve(VQ[0:mk,0:mk],VQ[0:mk,mk:mk+1])
            #VQ_lu = lu_Crout(VQ)
            #print(vq-VQ_lu[-1,-1])
            #print('*',VQ_lu[-1,-1])
            #vqs.append(abs(VQ_lu[-1,-1]))
            vqs.append(abs(vq))
            if vqs[-1]/vqs[0] < tao or cnt%mit == 0:
                restart = True
                mk = -1
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
                    H_tmp = np.linalg.inv(V_tmp.T @ P_tmp) @ (V_tmp.T @ (A @ P_tmp))
                    print('^', np.linalg.norm(Hk - H_tmp) / np.linalg.norm(H_tmp)) '''
                    VQ = Vk[:,0:mk+1].T@Qk[:,0:mk+1]                      
                
                if mk>0:
                    P_tmp = Pk[:,0:mk]
                    V_tmp = Vk[:,0:mk] 

                    H_tmp = np.linalg.inv(V_tmp.T @ P_tmp) @ (V_tmp.T @ (A @ P_tmp))                
                    #print(mk)
                    #print(Hk)
                    #print(H_tmp)
                    print('*', np.linalg.norm(Hk - H_tmp) / np.linalg.norm(H_tmp)) 
                    
                    err_linear = -Qk[:,mk-1]-Pk[:,0:mk+1]@Hk_bar[0:mk+1,mk-1]
                    err_linear = np.linalg.norm(err_linear,ord=2)
                    err_ratio = err_linear/np.linalg.norm(Qk[:,mk-1],ord=2)
                    #print('err:',err_linear,err_ratio)
                    assert err_linear<1e-8


        x_prev = xk.copy()
        r_prev = rk.copy()        

        mk = mk+1
        
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

        if cnt>0:
            relres_bar_v.append(norm2(rk) / nd)

        xk += beta * rk  # rk_bar = rk
        cnt = cnt + 1

        fk, gk = fh(xk)

        if cnt > 1:
            print(cnt - 1, "%e %e %e" % (fk, norm2(gk) / nd, norm2(rk) / nd))
            # print(relres_v)
    fk, gk = fh(xk)
    if type2:
	    filename = 'iram2.ser'
    else:
	    filename = 'iram1.ser'
    with open(filename, 'wb') as f:
        assert len(relres_v) == len(relres_bar_v)
        pickle.dump(relres_v, f)

    eig_tmp = np.linalg.eigvals(A)
    print('****************')
    # print(eig_tmp)

    '''import matplotlib.pyplot as plt
    print(len(eig_tmp), len(eig))
    plt.clf()
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='red', marker='o')
    plt.scatter(eig.real, eig.imag, label='IRAM', color='blue', marker='+')
    plt.legend()
    plt.show() '''
    import matplotlib
    import matplotlib.pyplot as plt
    plt.switch_backend('pgf')
    matplotlib.rcParams['font.family'] = 'Times New Roman' 
    print(len(eig_tmp), len(eig))
    plt.tick_params(labelsize='x-large')
    plt.xlabel('Real part',fontsize='xx-large')
    plt.ylabel('Imaginary part',fontsize='xx-large')
    plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='darkorange', marker='o')
    label2,color2,marker2 = ('IRAM-II','blue','+') if type2 else ('IRAM-I','purple','x')
    plt.scatter(eig.real, eig.imag, label=label2, color='purple', marker='+')
    plt.legend(fontsize='x-large')
    #plt.legend()
    #plt.show()
    filename2 = 'eigs_am2_linear.pdf' if type2 else 'eigs_am1_linear.pdf'
    plt.savefig(filename2,format='pdf',ppi=1200,bbox_inches='tight')
    #plt.savefig('figure_am2_linear.pdf',format='pdf',dpi=120,bbox_inches='tight') 
    print('iram method converge to a relative residual of %e, misfit value of %e in %d iterations.\n' % (
            norm2(gk) / nd, fk, cnt))  
             

        
#x0 = np.zeros((D_out,1))
#am2(misfit_total,x0,tol=1e-15,maxit=D_out+1000,tao=0,beta=0.001,m=25)

#x0 = np.zeros((D_out,1))
#anderson1(misfit_total,x0,tol=1e-6,maxit=D_out+1000,beta=0.001,m=26)

#x0 = np.zeros((D_out,1))
#anderson2(misfit_total,x0,tol=1e-6,maxit=D_out+1000,beta=0.001,m=26)

#x0 = np.zeros((D_out,1))
#anderson2(misfit_total,x0,tol=1e-6,maxit=D_out+1000,beta=0.001,m=1000)

#x0 = np.zeros((D_out,1))
#am2(misfit_total,x0,tol=1e-6,maxit=D_out+6000,beta=0.001,m=26)

#x0 = np.zeros((D_out,1))
#rv1 = gmres(misfit_total,x0,tol=1e-15,maxit=D_out+10)


#print(np.sort(np.linalg.eigvals(A)))

#x0 = np.zeros((D_out,1))
#iram(misfit_total,x0,tol=1e-6,maxit=D_out+1000,tao=0,beta=0.001,m=25,n=15,mit=1000,type2=False)
#iram(misfit_total,x0,tol=1e-6,maxit=2000,tao=0,beta=0.001,m=5,n=4,mit=1000,type2=True)

#eig_tmp = np.linalg.eigvals((A+A.T)/2)
#print(min(eig_tmp))

'''eig_tmp = np.linalg.eigvals(A)
print('****************')
import matplotlib
import matplotlib.pyplot as plt
#plt.switch_backend('pgf')
#matplotlib.rcParams['font.family'] = 'Times New Roman' 
print(len(eig_tmp), len(eig2))
plt.tick_params(labelsize='x-large')
plt.xlabel('Real part',fontsize='xx-large')
plt.ylabel('Imaginary part',fontsize='xx-large')
plt.scatter(eig_tmp.real, eig_tmp.imag, label='eigenvalue', color='darkorange', marker='o')
plt.scatter(eig2.real, eig2.imag, label='AM-II', color='purple', marker='+')
plt.legend(fontsize=13.6)
#plt.legend()
plt.show()
#plt.savefig('linear_figure_am1_linear.pdf',format='pdf',ppi=1200,bbox_inches='tight') '''

#x0 = np.zeros((D_out,1))
#rv1 = arnoldi(misfit_total,x0,tol=1e-15,maxit=D_out+10)


