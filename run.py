import scipy.integrate
import test

# Turn the Cython C function into a LowLevelCallable

"""# 'integrand' here is A*x*x + b
# - see test.pyx

A = 0.0
b = 0.0
res = scipy.integrate.quad(integrand, 0, 4, args=(A, b))
print('Should be 0: %g' % res[0])

A = 0.0
b = 1.0
res = scipy.integrate.quad(integrand, 0, 4, args=(A, b))
print('Should be 4: %g' % res[0])

A = 1.0
b = 0.0
res = scipy.integrate.quad(integrand, 0, 4, args=(A, b))
print('Should be 21.333...: %g' % res[0])"""
