#pragma once
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif
// BLAS
    float sdot_(const int *N, const float *X, const int *incX, const float *Y,
                const int *incY);
    double ddot_(const int *N, const double *X, const int *incX, const double *Y,
                 const int *incY);
    void cdotc_(std::complex<float> *result, const int *n, const std::complex<float> *zx,
                const int *incx, const std::complex<float> *zy, const int *incy);
    void cdotu_(std::complex<float> *result, const int *n, const std::complex<float> *zx,
                const int *incx, const std::complex<float> *zy, const int *incy);
    void zdotc_(std::complex<double> *result, const int *n, const std::complex<double> *zx,
                const int *incx, const std::complex<double> *zy, const int *incy);
    void zdotu_(std::complex<double> *result, const int *n, const std::complex<double> *zx,
                const int *incx, const std::complex<double> *zy, const int *incy);
    
    void sgemv_(const char *transa, const int *m, const int *n, const float *alpha,
                const float *a, const int *lda, const float *x, const int *incx,
                const float *beta, float *y, const int *incy);
    void dgemv_(const char *transa, const int *m, const int *n, const double *alpha,
                const double *a, const int *lda, const double *x, const int *incx,
                const double *beta, double *y, const int *incy);
    void cgemv_(const char *transa, const int *m, const int *n,
                const std::complex<float> *alpha, const std::complex<float> *a,
                const int *lda, const std::complex<float> *x, const int *incx,
                const std::complex<float> *beta, std::complex<float> *y, const int *incy);
    void zgemv_(const char *transa, const int *m, const int *n,
                const std::complex<double> *alpha, const std::complex<double> *a,
                const int *lda, const std::complex<double> *x, const int *incx,
                const std::complex<double> *beta, std::complex<double> *y, const int *incy);
    
    void sgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const float *alpha, const float *a, const int *lda,
                const float *b, const int *ldb, const float *beta, float *c,
                const int *ldc);
    void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const double *alpha, const double *a, const int *lda,
                const double *b, const int *ldb, const double *beta, double *c,
                const int *ldc);
    void cgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const std::complex<float> *alpha, const std::complex<float> *a,
                const int *lda, const std::complex<float> *b, const int *ldb,
                const std::complex<float> *beta, std::complex<float> *c, const int *ldc);
    void zgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const std::complex<double> *alpha,
                const std::complex<double> *a, const int *lda, const std::complex<double> *b,
                const int *ldb, const std::complex<double> *beta, std::complex<double> *c,
                const int *ldc);
    void dzgemm_(const char *transa, const char *transb, const int *m, const int *n,
                 const int *k, const std::complex<double> *alpha, const double *a,
                 const int *lda, const std::complex<double> *b, const int *ldb,
                 const std::complex<double> *beta, std::complex<double> *c, const int *ldc);
// LAPACK
    void sgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
                const int *lda, float *wr, float *wi, float *vl, const int *ldvl,
                float *vr, const int *ldvr, float *work, const int *lwork, int *info);
    void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
                const int *lda, double *wr, double *wi, double *vl, const int *ldvl,
                double *vr, const int *ldvr, double *work, const int *lwork, int *info);
    void cgeev_(const char *jobvl, const char *jobvr, const int *n, float *a,
                const int *lda, std::complex<float> *wr, std::complex<float> *wi, std::complex<float> *vl, const int *ldvl,
                std::complex<float> *vr, const int *ldvr, std::complex<float> *work, const int *lwork, int *info);
    void zgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
                const int *lda, std::complex<double> *wr, std::complex<double> *wi, std::complex<double> *vl, const int *ldvl,
                std::complex<double> *vr, const int *ldvr, std::complex<double> *work, const int *lwork, int *info);

    void ssyev_(const char *jobz, const char *uplo, const int *n, float *a,
                const int *lda, float *w, float *work, const int *lwork,
                int *info);
    void dsyev_(const char *jobz, const char *uplo, const int *n, double *a,
                const int *lda, double *w, double *work, const int *lwork,
                int *info);
    void cheev_(const char *jobz, const char *uplo, const int *n,
                std::complex<float> *a, const int *lda, float *w,
                std::complex<float> *work, const int *lwork, float *rwork, int *info);
    void zheev_(const char *jobz, const char *uplo, const int *n,
                std::complex<double> *a, const int *lda, double *w,
                std::complex<double> *work, const int *lwork, double *rwork, int *info);

    void sgetrf_(const int *m, const int *n, float *A, const int *lda, int *ipiv, int *info);
    void dgetrf_(const int *m, const int *n, double *A, const int *lda, int *ipiv, int *info);
    void cgetrf_(const int *m, const int *n, std::complex<float> *A, const int *lda, int *ipiv, int *info);
    void zgetrf_(const int *m, const int *n, std::complex<double> *A, const int *lda, int *ipiv, int *info);
                           
    void sgetri_(const int *n, float *A, const int *lda, int *ipiv, float *work, const int *lwork, int *info);
    void dgetri_(const int *n, double *A, const int *lda, int *ipiv, double *work, const int *lwork, int *info);
    void cgetri_(const int *n, std::complex<float> *A, const int *lda, int *ipiv, std::complex<float> *work, const int *lwork, int *info);
    void zgetri_(const int *n, std::complex<double> *A, const int *lda, int *ipiv, std::complex<double> *work, const int *lwork, int *info);
#ifdef __cplusplus
}
#endif
