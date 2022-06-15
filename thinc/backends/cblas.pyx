# distutils: sources = thinc/backends/cblas_impl.cpp

cimport blis.cy

cdef class CBlas:
    def __init__(self):
        """Construct a CBlas instance set to use BLIS implementations of the
           supported BLAS functions."""
        self.c_impl.set_saxpy(blis.cy.saxpy)
        self.c_impl.set_sgemm(blis.cy.sgemm)

    cdef CBlasImpl c(self) nogil:
        return self.c_impl
