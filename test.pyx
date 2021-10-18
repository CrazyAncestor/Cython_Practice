import scipy.integrate as integrate
import numpy as np
# The integrand here is A*x*x + b

# NB: this has to be a cdef, not a cpdef or def, because these add
# extra stuff to the argument list to help python. LowLevelCallable
# does not like these things...

# You can however increase the number of arguments (remember also to
# update test.pxd)
cpdef double integrand(double x):
     return x*x

cpdef double integration(double x,int n):
     for i in range(n):
          integrate.quad(integrand,0,x)
     return integrate.quad(integrand,0,x)[0]

cpdef double f(double x, double y):
     return x*y

cpdef double integration_2nd(double t,int n):
     for i in range(n):
          integrate.nquad(f, [[0,1], [0,1]])
     return integrate.nquad(f, [[0,1], [0,1]])[0]

cpdef double g(double alpha, double Ev, double mx):
     cdef double gamma = (Ev + mx)/((mx**2+2*mx*Ev)**0.5)
     
     cdef double beta = 1/(1-gamma**(-2))**(-0.5)
     cdef double sec = 1/np.cos(alpha)

     # Artificial Criterium 1
     # Avoid Backward Scattering

     if alpha<np.pi/2: 
          return sec**3*2*(1-beta**2)/(sec**2-beta**2)**2
     else:
          return 0

cpdef double dEvdTx(double Tx,double mx,double alpha):
    """
    Calculate dEv/dTx via analytical expression
    
    Input
    ------
    Tx: DM kinetic energy
    mx: DM mass
    alpha: scattering angle in lab frame
    dTx: arbitrarity small number, default 1e-5
    
    Output
    ------
    tuple: dEv/dTx
    """
    if alpha==0:
        return 1+0.25*(Tx**2+2*Tx*mx)**(-0.5)*(Tx*2+2*mx)

    cdef double v1 = 4*mx*Tx + 8*mx*Tx*np.tan(alpha)**(-2)
    cdef double v2 = np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2))
    cdef double v3 = 2*Tx + np.sqrt(2*Tx*(2*mx + Tx + 4*mx*np.tan(alpha)**(-2) + v2))
    cdef double v4 = 8*Tx*v2
    return (v1 + (Tx + v2)*v3)/v4

cpdef rho(double rho_s,double x):
     return rho_s/(x*(1+x)**2)

cpdef fv(double Ev,double Tnu):
     # Fermi-Dirac distribution
     return (1/18.9686)*(1/Tnu**3)*(Ev**2/(np.exp(Ev/Tnu - 3)+1))

cpdef f_Ev(Ev):
    return fv(Ev,2.76)/11 +fv(Ev,4.01)/16 + 4*fv(Ev,6.26)/25

cpdef _Ev(double Tx,double mx,double alpha):
    """
    Calculate the neutrino energy to produce DM kinetic energy at
    lab frame scattering angle alpha via analytical expression
    
    Input
    ------
    Tx: DM kinetic energy
    mx: DM mass
    alpha: scattering angle in lab frame
    
    Output
    Ev: the corresponding neutrino energy
    """
    if alpha==0:
        return Tx+0.5*(Tx**2+2*Tx*mx)**0.5
    
    return  -0.5*mx + Tx + np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2)) \
            + np.sqrt(2*Tx*(2*mx + Tx + 4*mx*np.tan(alpha)**(-2)+   \
                    + np.sqrt((2*mx + Tx)**2 + 8*mx*Tx*np.tan(alpha)**(-2))))

cpdef fl(double tx,double l,double theta,mx): 
     cdef double beta = (tx*(tx + 2*mx))**0.5/(tx+mx)
     cdef double cos = np.cos(theta)
     cdef double r = (l**2 + 1. - 2 *l *1.0*cos)**0.5
     cdef double alpha = np.arcsin(np.sin(theta)/r)
     
     if np.isnan(alpha) or (r<(0.03)):
         return 0.
 
     # Geometric Terms
     cdef double geometric_terms = np.sin(theta) *2*np.pi
     
     # Physical terms
     cdef double Ev = _Ev(tx,mx,alpha)
     if Ev<=0 or np.isnan(Ev):
         return 0 
     cdef double dn_domega= g(alpha,Ev,mx)
     cdef double dEv_dTx = dEvdTx(tx,mx,alpha)
     
     cdef double physical_terms = rho(0.184e3,r/3)/r**2*dn_domega*dEv_dTx*f_Ev(Ev) 
     return geometric_terms*physical_terms