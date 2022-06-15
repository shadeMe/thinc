#include "cblas_impl.hh"

struct BlasFuncs {
  daxpy_ptr daxpy;
  saxpy_ptr saxpy;
  sgemm_ptr sgemm;
};

CBlasImpl::CBlasImpl() {
  blas_funcs.reset(new BlasFuncs);
}

daxpy_ptr CBlasImpl::daxpy() {
  return blas_funcs->daxpy;
}

saxpy_ptr CBlasImpl::saxpy() {
  return blas_funcs->saxpy;
}

sgemm_ptr CBlasImpl::sgemm() {
  return blas_funcs->sgemm;
}

void CBlasImpl::set_daxpy(daxpy_ptr daxpy) {
  blas_funcs->daxpy = daxpy;
}

void CBlasImpl::set_saxpy(saxpy_ptr saxpy) {
  blas_funcs->saxpy = saxpy;
}

void CBlasImpl::set_sgemm(sgemm_ptr sgemm) {
  blas_funcs->sgemm = sgemm;
}
