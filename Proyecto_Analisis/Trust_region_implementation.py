from cmath import pi
from pickle import TRUE
from re import X
import numpy as np
import scipy.linalg as LA
import scipy.optimize as OP

from Gra_Jac_Hess import apGrad,apHess
tol=10e-5
itmax=1000
Delta_max=1.5
eta=0.1

def pDogLeg(B,g,Delta):
    p_u=((-1)*((g.T@g)/(g.T@B@g)))*g
    if (p_u.T@p_u)>=Delta**2:
        p=-(Delta/((g.T@g)**(1/2)))*(g.copy())
    else:
        p_B=-LA.inv(B)@g
        if (p_B.T@p_B)<=Delta**2:
            p=p_B
            #print(p)
        else:
            coeff=[p_B.T@p_B,2*((p_B-p_u).T@p_u),(p_u.T@p_u)-Delta**2]
            alpha=np.roots(coeff)
            for i in alpha:
                if 0<i<1:
                    alpha_s=i
                    #print(alpha_s)
            p= p_u + alpha_s*(p_B-p_u)      
    return p

def pCauchy(B, g , Delta):
    dummy_v1=g.T@ B @ g
    ps_k=-(Delta/((g.T@g)**(1/2)))*(g.copy())
    if dummy_v1<=0:
        tao_k=1
    else:
        dummy_v2= ((g.T@g)**(3/2))/(Delta*dummy_v1)
        tao_k=min(dummy_v2,1)
    pC=tao_k*ps_k
    return pC


def mRC1( f, x0, itmax ):
    k=0
    Delta=Delta_max
    x=x0.copy()
    n=len(x)
    m_k=lambda x,g,p,B,: f(x) +g.T@p+(0.5)*p.T@B@p
    for k in range(itmax):
        gk=apGrad(f,x)
        print(gk)
        if np.linalg.norm(gk,ord=np.inf,axis=0) <= tol:
            print('Happy iteration',k)
            break
        else:
            Bk=apHess(f,x)
            
            dk=pCauchy(Bk,gk,Delta)
            p_k=(f(x)-f(x+dk))/(m_k(x,gk,np.zeros(n),Bk)-m_k(x,gk,dk,Bk))
            if p_k<0.25:
                Delta=0.25*Delta
            else:
                if p_k>0.75 and (dk.T@dk)==Delta**2:
                    Delta=min(2*Delta,Delta_max)
            if p_k>eta:
                x=x+dk

    return x

def mRC2( f, x0, itmax ):
    k=0
    Delta=Delta_max
    x=x0.copy()
    n=len(x)
    s=lambda c:(10**-12)-(9/8)*c
    m_k=lambda x,g,p,B,: f(x) +g.T@p+(0.5)*p.T@B@p
    for k in range(itmax):
        gk=apGrad(f,x)
        if np.linalg.norm(gk,ord=np.inf,axis=0) <= tol:
            print('Happy iteration',k)
            break
        else:
            Bk=apHess(f,x)
            lam=LA.eigh(Bk,eigvals_only=True).min()
            if lam<=0:
                Bk=Bk+s(lam)*np.eye(n)
            dk=pDogLeg(Bk,gk,Delta)
            p_k=(f(x)-f(x+dk))/(m_k(x,gk,np.zeros(n),Bk)-m_k(x,gk,dk,Bk))
            if p_k<0.25:
                Delta=0.25*Delta
            else:
                if p_k>0.75 and (dk.T@dk)==Delta**2:
                    Delta=min(2*Delta,Delta_max)
            if p_k>eta:
                x=x+dk

    return x


f= lambda x: np.array((x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2)
g= lambda x: np.array((1+((x[0]+x[1]+1)**2)*(19-14*x[0]+3*(x[0]**2)-14*x[1]+6*x[0]*x[1]+3*(x[1]**2)))*(30+((2*x[0]-3*x[1])**2)*(18-32*x[0]+(12*x[0]**2)+48*x[1]-36*x[0]*x[1]+27*(x[1]**2))))
h=lambda x:np.array(0.26*((x[0]**2)+(x[1]**2))-0.48*x[0]*x[1])
w= lambda x:np.array(-np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0]-np.pi)**2)-(x[1]-np.pi)**2))
t=lambda x:np.array(2*x[0]**2-1.05*x[0]**4+(1/6)*x[0]**6+x[0]*x[1]+x[1]**2)
x_0=np.array([np.pi-0.5,np.pi-0.5])
print(w([np.pi,np.pi]))
#pC1=pCauchy(apHess(f,x_0),apGrad(f,x_0),2)
#pC2=pDogLeg(apHess(f,x_0),apGrad(f,x_0),2)
#print(pC1,pC2)
#print(apGrad(f,x_0))
#print(apHess(f,x_0))

#m_k=lambda p,g,B,x: f(x) +g.T@p+(0.5)*p.T@B@p
#print(m_k(apGrad(f,x_0),apGrad(f,x_0),apHess(f,x_0),x_0))
#print(f(x_0))

print(mRC1( f, x_0, itmax ))
print(mRC2( f, x_0, itmax ))
#print(np.eye(2))
#print(apHess(w,x_0))
#print(apGrad(w,x_0))
#OP.minimize(f, x_0, method='dogleg', jac=TRUE)