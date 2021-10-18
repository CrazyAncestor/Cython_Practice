# The integrand here is A*x*x + b

# NB: this has to be a cdef, not a cpdef or def, because these add
# extra stuff to the argument list to help python. LowLevelCallable
# does not like these things...

# You can however increase the number of arguments (remember also to
# update test.pxd)
cdef double integrand(int n, double[3] args):

     # x-coordinate
     x = args[0]

     # coefficients
     A = args[1]

     # coefficients
     b = args[2]

     return A*x*x + b
     
