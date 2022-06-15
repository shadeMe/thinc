#ifndef CBLAS_HH
#define CBLAS_HH

#include <memory>

typedef int bint;

struct BlasFuncs;

typedef void (*sgemm_ptr)(bint transA, bint transB, int M, int N, int K,
                          float alpha, const float* A, int lda, const float *B,
                          int ldb, float beta, float* C, int ldc);


typedef void (*daxpy_ptr)(int N, double alpha, const double* X, int incX,
                          double *Y, int incY);


typedef void (*saxpy_ptr)(int N, float alpha, const float* X, int incX,
                          float *Y, int incY);


class CBlasImpl {
public:
  CBlasImpl();
  virtual ~CBlasImpl() {}

  daxpy_ptr daxpy();
  saxpy_ptr saxpy();
  sgemm_ptr sgemm();
  void set_daxpy(daxpy_ptr daxpy);
  void set_saxpy(saxpy_ptr saxpy);
  void set_sgemm(sgemm_ptr sgemm);

private:
  std::shared_ptr<BlasFuncs> blas_funcs;
};



#endif // CBLAS_HH
