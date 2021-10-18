import scipy.integrate as integrate
import test

# Turn the Cython C function into a LowLevelCallable
print(test.integration_2nd(1,40000))

def f(x, y):
     return x*y

def integration_2nd(t,n):
     for i in range(n):
          integrate.nquad(f, [[0,1], [0,1]])
     return integrate.nquad(f, [[0,1], [0,1]])[0]
#print(integration_2nd(1,40000))


