from libcpp.memory cimport shared_ptr


ctypedef void (*sgemm_ptr)(bint transA, bint transB, int M, int N, int K,
                           float alpha, const float* A, int lda, const float *B,
                           int ldb, float beta, float* C, int ldc) nogil


ctypedef void (*daxpy_ptr)(int N, double alpha, const double* X, int incX,
                           double *Y, int incY) nogil


ctypedef void (*saxpy_ptr)(int N, float alpha, const float* X, int incX,
                           float *Y, int incY) nogil


# Forward-declaration of the BlasFuncs struct. This struct must be opaque, so
# that consumers of the CBlas class cannot become dependent on its size or
# ordering.
cdef struct BlasFuncs



cdef extern from "cblas_impl.hh":
    cdef cppclass CBlasImpl:
        CBlas() nogil
        daxpy_ptr daxpy() nogil
        saxpy_ptr saxpy() nogil
        sgemm_ptr sgemm() nogil
        void set_daxpy(daxpy_ptr daxpy) nogil
        void set_saxpy(saxpy_ptr saxpy) nogil
        void set_sgemm(sgemm_ptr sgemm) nogil


cdef class CBlas:
    cdef CBlasImpl c_impl
    cdef CBlasImpl c(self) nogil
