*NB: see also the corresponding article on my homepage: [faster
numerical integration in
SciPy](https://saoghal.net/articles/2020/faster-numerical-integration-in-scipy/),
which has basically the same content as this README.*

Calling Cython from SciPy
=========================

Perhaps you have noticed that SciPy offers [low-level callback
functions](https://docs.scipy.org/doc/scipy/reference/ccallback.html)
that let you use compiled code (e.g. from C, or via Numba or Cython)
as, for example, the integrand in
[`scipy.integrate.quad()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad).

I was keen to help a colleague accelerate their numerical
integrations, while keeping the project as high-level and pythonic as
possible. Faced with the above options (C, Numba, Cython), I noticed
that none of them were documented particularly well, but Cython was
the one that seemed likely to result in the cleanest, most modular
code. It also happens that I was familiar with Cython.

It took me a couple of hours to figure out how to get this to work
properly, and with no real tutorial or simple advice to follow, I made
[a starter kit](https://version.helsinki.fi/weir/cython-and-scipy)
showing a minimal working example.

Note that I don't fully know all the jargon names given to bits of
Cython. Frankly, I didn't find all the answers I needed in the Cython
documentation and had to do some good old-fashioned guesswork and
piecing things together. See the _further reading_ below.


How does it work
----------------

Long story short,
[`LowLevelCallable.from_cython()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.from_cython.html#scipy.LowLevelCallable.from_cython)
needs to be pointed at a compiled, C-style function to work. You can
do that with C, Cython, or Numba. Of these, the Cython method seems
the easiest, but is strangely the least documented.

The starter kit contains the following files:

-   `test.pyx` is a Cython file, which contains a C-style function
    called `integrand()`, defined using `cdef` (see Cython's [language
    basics](https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html)
    for an explanation of what `cdef` is). We want to turn this into
    C, compile it, and then later on use it to instantiate a
    [`scipy.LowLevelCallable`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.LowLevelCallable.html)
    instance so `scipy.integrate` can use that rather than working
    with interpreted python (which is slow).
	
-   `test.pxd` is a Cython "header" file (I don't know what the
    correct name is, but the [Cython
    documentation](https://cython.readthedocs.io/en/latest/src/tutorial/pxd_files.html)
    uses the same analogy). When (I think) `cythonize()` in `setup.py`
    detects this, it knows it's building a Cython file we want to call
    from C and it should behave itself accordingly, exporting symbols
    that a C linker can use if need be. Otherwise, Cython does not let
    c-style functions (those defined with cdef) be called from outside
    the Cython file, e.g. from other C code.

    The corresponding python module then has the attribute
    `__pyx_capi__`, which is also needed for
    `LowLevelCallable.from_cython()` to work. To be honest, I couldn't
    find useful documentation about this, but I think this means that
    it has a 'parallel' C interface, as it were.

    [This](https://github.com/ashwinvis/cython_capi/tree/master/using_pxd)
    subdirectory of a repo by Ashwin Vishnu was useful, as was [this
    article](http://pdebuyl.be/blog/2017/cython-module.html) by Pierre
    de Buyl.

-   `setup.py` is the usual distutils thing, but it knows it's
    building a Cython module because of the call to `cythonize()` and
    thus detects the `.pxd` file (there is otherwise no relationship
    between them, no `cimport` or similar); I'm not entirely
    sure, though.

-   `run.py` uses the `integrand()` function from `test.pyx` to do
    some numerical integrals.


To try this out you will need: python 3, cython

1.  Run:

        $ python3 setup.py build_ext --inplace

    This builds the module from test.pyx (with symbols from
    test.pxd). Although running setup.py like this does the
    compilation for you, you can by the way see the C code generated
    in test.c, the function call starts something like:

        static double __pyx_f_4test_integrand(CYTHON_UNUSED int __pyx_v_n, double *__pyx_v_args)

    [`CYTHON_UNUSED` _I think_ because the argument n is not used in `integrand()`, but the standard rubric for `LowLevelCallable.from_cython()` needs it]

    But note that it really churns out `double`s, not some wacky
    higher-level type. This is really C at the end of the day.

2.  To test:

        $ python3 run.py
        Should be 0: 0
        Should be 4: 4
        Should be 21.333...: 21.3333

    Uses
    [`scipy.integrate.quad()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html#scipy.integrate.quad)
    to evaluate three numerical integrals. Note that I've tried to
    make it clear how to pass numerical coefficients from the python
    to the C code, so things don't need to be hardcoded.


Outstanding issues
------------------

I didn't quite figure out how to make this work with NumPy data
types. But maybe that doesn't matter.



Update 7.4.2020: some tips on getting Cython to generate native C code
----------------------------------------------------------------------

1.  If you employ float division, use the Cython decorator
    `@cython.cdivision(True)` before a `cdef` function:

        import cython
		
		...

        # The decorator stops division by zero checking
        @cython.cdivision(True)
		cdef double C1(double t, double t_cut, double t_0):

            ... etc ...

    to avoid Python checking for a `DivisionByZeroError`. Note that
    you will need `import cython` to get access to this decorator.
	
2.  You can find [Cython version of Scipy special
    functions](https://docs.scipy.org/doc/scipy/reference/special.cython_special.html)
    in `scipy.special.cython_special`, for the Bessel function
    normally given by [`scipy.special.jv`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html)
    we needed to do:
	
        from scipy.special.cython_special cimport jv

3.  Similarly, Cython makes available C versions of many common C
    library functions ([read more
    here](https://cython.readthedocs.io/en/latest/src/tutorial/external.html)):

        from libc.math cimport sqrt, exp, sin, cos
		
4.  Note that although Cython knows about the C99 complex numbers, I
    still haven't quite worked out how to do fast mathematics with
    them directly (but using the real and imaginary parts separately
    is ok).
	
5.  Keep checking the yellow-highlighted 'annotation' HTML file Cython
    generates, and don't panic if something is still yellow (meaning
    it may be calling interpreted Python still). Click the `+` to the
    left of the line number to see what is actually happening.
	
	Even after doing the above things, we were using 
	
	    :::python
	    return result.real
	
	which led to some yellow highlighting. But the underlying code
    contained the following preprocessor stuff:
	

		#if CYTHON_CCOMPLEX
		  #ifdef __cplusplus
			#define __Pyx_CREAL(z) ((z).real())
			#define __Pyx_CIMAG(z) ((z).imag())
		  #else
			#define __Pyx_CREAL(z) (__real__(z))
			#define __Pyx_CIMAG(z) (__imag__(z))
		  #endif
		#else
			#define __Pyx_CREAL(z) ((z).real)
			#define __Pyx_CIMAG(z) ((z).imag)
		#endif
	
	Comparing this with [the GCC
    documentation](http://gcc.gnu.org/onlinedocs/gcc/Complex.html), it
    seems we are generating native C code after all, but Cython can't
    tell that.

Further reading
---------------


*   [Vegas documentation - faster
    integrands](https://vegas.readthedocs.io/en/latest/tutorial.html#faster-integrands) -
    my collaborator Pierre Auclair pointed me to this description in
    the documentation for Vegas which has some similar advice.

*   [Developing a Cython
    library](http://pdebuyl.be/blog/2017/cython-module.html) by Pierre
    de Buyl helped me figure out that I needed a `.pxd` file.

*   [Experiments with Cython C API and
    PyCapsules](https://github.com/ashwinvis/cython_capi) by Ashwin
    Vishnu is also worth glancing at, but also has some valuable
    references to other similar examples of interfacing Cython, Python
    and C.

*   [SciPy's new LowLevelCallable is a
    game-changer](https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/)
    by Juan Nunez-Iglesias is written from the point of view of Numba
    and image processing (`ndimage`), but is useful too.

*   [Passing Numpy arrays to C code wrapped with
    Cython](https://stackoverflow.com/questions/4495420/passing-numpy-arrays-to-c-code-wrapped-with-cython)
    is a StackOverflow question of potential reference to getting
    NumPy to play along.

*   [Introduction to Cython for Solving Differential
    Equations](http://hplgit.github.io/teamods/cyode/main_cyode.html)
    by Hans Petter Langtangen does not cover the
    scipy.LowLevelCallable integration, but has lots of other useful
    tidbits.
