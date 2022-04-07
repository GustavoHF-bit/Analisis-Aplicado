#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approximate gradients and hessians with finite differences.
Created on Tue Oct 12 17:42:00 2021

@author: Andreas Wachtel
"""


import numpy as np
import numpy.testing as test
import scipy.linalg as LA


def apGrad(f, x):
    assert( type(x) is np.ndarray ), 'x debe ser numpy.array'
    
    if len(x.shape)==2:
        x=x[:,0]
    
    n=len(x)
    #Id=np.eye(n)
    #np.finfo(float).eps
    hs=np.diag((np.finfo(float).eps**(1.0/3))*(abs(x)+1))
    
    g=np.zeros(n)
    for i in range(n):
        g[i]=0.5*(f(x+hs[:,i])-f(x-hs[:,i]))/hs[i,i]
    return g



def apJacobian(f, x):
    """f: Rn->Rm"""
    assert( type(x) is np.ndarray ), 'x debe ser numpy.array'
    
    if len(x.shape)==2: 
        x=x[:,0]
    
    n=len(x)
    m=len(f(x))
    #Id=np.eye(n)
    #np.finfo(float).eps
    hs=np.diag((np.finfo(float).eps**(1.0/3))*(abs(x)+1))
    
    jac=np.zeros((m,n))
    for j in range(n):
        jac[:,j]=0.5*(f(x+hs[:,j])-f(x-hs[:,j]))/hs[j,j]
    
    
                
    return jac



def apHess(f, x):
    assert( type(x) is np.ndarray ), 'x debe ser numpy.array'
     
    if len(x.shape)==2:
            x=x[:,0]
    
    n=len(x)
    #Id=np.eye(n)
    #np.finfo(float).eps
    hs=np.diag((np.finfo(float).eps**(1.0/4))*(abs(x)+1))
    
    Hf=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Hf[i,j]=0.25*(f(x+hs[:,i]+hs[:,j])\
                          -f(x-hs[:,i]+hs[:,j])\
                          -f(x+hs[:,i]-hs[:,j])\
                          +f(x-hs[:,i]-hs[:,j]))/(hs[i,i]*hs[j,j])
    return Hf




def test1():
    A = LA.pascal(4).astype(float)
    b = -np.ones(4)
    f  = lambda x:  0.5*np.dot(x, A@x) + np.dot(b,x)
    Df = lambda x:  0.5*(A+A.T)@x + b
    
    x  = np.ones(4)
    g  = apGrad(f, x)
    ge = Df(x)
    test.assert_almost_equal(g, ge)
    H  = apHess(f, x)
    test.assert_almost_equal(H, A)




if __name__ == '__main__':
    test1()
    print("This file was executed from the command line or an interpreter.")
else:
    print("Imported: Finite Difference Gradient and Hessian (by AW).")
