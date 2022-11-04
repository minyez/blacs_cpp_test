#pragma once
#include "interfaces/blaslapack.h"
#include "interfaces/blacsscalapack.h"
#include <stdexcept>
#include <string.h>

namespace linalg
{
    static inline char revert_trans(const char &trans)
    {
        switch (trans) {
            case 'T':
                return 'N';
            case 't':
                return 'n';
            case 'N':
                return 'T';
            case 'n':
                return 't';
            case 'C':
                throw std::invalid_argument("does not support C, require manual handling");
            case 'c':
                throw std::invalid_argument("does not support C, require manual handling");
            default:
                throw std::invalid_argument("invalid trans character");
        }
    }

    static inline char revert_uplo(const char &uplo)
    {
        switch (uplo) {
            case 'U':
                return 'L';
            case 'u':
                return 'l';
            case 'L':
                return 'U';
            case 'l':
                return 'u';
            default:
                throw std::invalid_argument("invalid uplo character");
        }
    }

    template <typename T>
    inline T* transpose(const T* a, const int &n, const int &lda, bool conjugate = false)
    {
        T* a_fort = new T[lda*n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < lda; j++)
                a_fort[i*lda+j] = a[j*n+i];
        return a_fort;
    }

    template <>
    inline std::complex<float>* transpose(const std::complex<float>* a, const int &n, const int &lda, bool conjugate)
    {
        std::complex<float>* a_fort = new std::complex<float>[lda*n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < lda; j++)
            {
                if (conjugate)
                    a_fort[i*lda+j] = conj(a[j*n+i]);
                else
                    a_fort[i*lda+j] = a[j*n+i];
            }
        return a_fort;
    }

    template <>
    inline std::complex<double>* transpose(const std::complex<double>* a, const int &n, const int &lda, bool conjugate)
    {
        std::complex<double>* a_fort = new std::complex<double>[lda*n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < lda; j++)
            {
                if (conjugate)
                    a_fort[i*lda+j] = conj(a[j*n+i]);
                else
                    a_fort[i*lda+j] = a[j*n+i];
            }
        return a_fort;
    }

    template <typename T>
    inline void transpose(const T* a_fort, T* a, const int &n, const int &lda, bool conjugate = false)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < lda; j++)
                a[j*n+i] = a_fort[i*lda+j];
    }

    template <>
    inline void transpose(const std::complex<float>* a_fort, std::complex<float>* a, const int &n, const int &lda, bool conjugate)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < lda; j++)
                a[j*n+i] = a_fort[i*lda+j];
    }

    template <>
    inline void transpose(const std::complex<double>* a_fort, std::complex<double>* a, const int &n, const int &lda, bool conjugate)
    {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < lda; j++)
                a[j*n+i] = a_fort[i*lda+j];
    }

    inline float dot(const int &N, const float *X, const int &incX, const float *Y, const int &incY)
    {
        return sdot_(&N, X, &incX, Y, &incY);
    }
    inline double dot(const int &N, const double *X, const int &incX, const double *Y, const int &incY)
    {
        return ddot_(&N, X, &incX, Y, &incY);
    }

    inline std::complex<float> dot(const int &N, const std::complex<float> *X, const int &incX,
                              const std::complex<float> *Y, const int &incY)
    {
        std::complex<float> res;
        cdotu_(&res, &N, X, &incX, Y, &incY);
        return res;
    }

    inline std::complex<double> dot(const int &N, const std::complex<double> *X, const int &incX,
                               const std::complex<double> *Y, const int &incY)
    {
        std::complex<double> res;
        zdotu_(&res, &N, X, &incX, Y, &incY);
        return res;
    }

    inline std::complex<float> dotc(const int &N, const std::complex<float> *X, const int &incX,
                               const std::complex<float> *Y, const int &incY)
    {
        std::complex<float> res;
        cdotc_(&res, &N, X, &incX, Y, &incY);
        return res;
    }

    inline std::complex<double> dotc(const int &N, const std::complex<double> *X, const int &incX,
                                const std::complex<double> *Y, const int &incY)
    {
        std::complex<double> res;
        zdotc_(&res, &N, X, &incX, Y, &incY);
        return res;
    }

    inline void gemm(const char &transa, const char &transb, const int &m, const int &n,
                     const int &k, const float &alpha, const float *a, const int &lda,
                     const float *b, const int &ldb, const float &beta, float *c, const int &ldc)
    {
        sgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
    }

    inline void gemm(const char &transa, const char &transb, const int &m, const int &n,
                     const int &k, const double &alpha, const double *a, const int &lda,
                     const double *b, const int &ldb, const double &beta, double *c, const int &ldc)
    {
        dgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
    }

    inline void gemm(const char &transa, const char &transb, const int &m, const int &n,
                     const int &k, const std::complex<float> &alpha, const std::complex<float> *a, const int &lda,
                     const std::complex<float> *b, const int &ldb, const std::complex<float> &beta, std::complex<float> *c, const int &ldc)
    {
        cgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
    }

    inline void gemm(const char &transa, const char &transb, const int &m, const int &n,
                     const int &k, const std::complex<double> &alpha, const std::complex<double> *a, const int &lda,
                     const std::complex<double> *b, const int &ldb, const std::complex<double> &beta, std::complex<double> *c, const int &ldc)
    {
        zgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
    }

    inline void gemm_f(const char &transa, const char &transb, const int &m, const int &n,
                       const int &k, const float &alpha, const float *a, const int &lda,
                       const float *b, const int &ldb, const float &beta, float *c, const int &ldc)
    {
        sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    inline void gemm_f(const char &transa, const char &transb, const int &m, const int &n,
                       const int &k, const double &alpha, const double *a, const int &lda,
                       const double *b, const int &ldb, const double &beta, double *c, const int &ldc)
    {
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    inline void gemm_f(const char &transa, const char &transb, const int &m, const int &n,
                       const int &k, const std::complex<float> &alpha, const std::complex<float> *a, const int &lda,
                       const std::complex<float> *b, const int &ldb, const std::complex<float> &beta, std::complex<float> *c, const int &ldc)
    {
        cgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    inline void gemm_f(const char &transa, const char &transb, const int &m, const int &n,
                       const int &k, const std::complex<double> &alpha, const std::complex<double> *a, const int &lda,
                       const std::complex<double> *b, const int &ldb, const std::complex<double> &beta, std::complex<double> *c, const int &ldc)
    {
        zgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }

    inline void gemv(const char &transa, const int &m, const int &n,
                     const float &alpha, const float *a, const int &lda,
                     const float *x, const int &incx, const float &beta, float *y, const int &incy)
    {
        char transa_f = revert_trans(transa);
        sgemv_(&transa_f, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv(const char &transa, const int &m, const int &n,
                     const double &alpha, const double *a, const int &lda,
                     const double *x, const int &incx, const double &beta, double *y, const int &incy)
    {
        char transa_f = revert_trans(transa);
        dgemv_(&transa_f, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv(const char &transa, const int &m, const int &n,
                     const std::complex<float> &alpha, const std::complex<float> *a, const int &lda,
                     const std::complex<float> *x, const int &incx, const std::complex<float> &beta, std::complex<float> *y, const int &incy)
    {
        char transa_f = revert_trans(transa);
        cgemv_(&transa_f, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv(const char &transa, const int &m, const int &n,
                     const std::complex<double> &alpha, const std::complex<double> *a, const int &lda,
                     const std::complex<double> *x, const int &incx, const std::complex<double> &beta, std::complex<double> *y, const int &incy)
    {
        char transa_f = revert_trans(transa);
        zgemv_(&transa_f, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }


    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const float &alpha, const float *a, const int &lda,
                       const float *x, const int &incx, const float &beta, float *y, const int &incy)
    {
        sgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const double &alpha, const double *a, const int &lda,
                       const double *x, const int &incx, const double &beta, double *y, const int &incy)
    {
        dgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const std::complex<float> &alpha, const std::complex<float> *a, const int &lda,
                       const std::complex<float> *x, const int &incx, const std::complex<float> &beta, std::complex<float> *y, const int &incy)
    {
        cgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const std::complex<double> &alpha, const std::complex<double> *a, const int &lda,
                       const std::complex<double> *x, const int &incx, const std::complex<double> &beta, std::complex<double> *y, const int &incy)
    {
        zgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void getrf(const int &m, const int &n, float *A, const int &lda, int *ipiv, int &info)
    {
        float *a_fort = transpose(A, n, lda);
        sgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getrf(const int &m, const int &n, double *A, const int &lda, int *ipiv, int &info)
    {
        double *a_fort = transpose(A, n, lda);
        dgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getrf(const int &m, const int &n, std::complex<float> *A, const int &lda, int *ipiv, int &info)
    {
        std::complex<float> *a_fort = transpose(A, n, lda);
        cgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getrf(const int &m, const int &n, std::complex<double> *A, const int &lda, int *ipiv, int &info)
    {
        std::complex<double> *a_fort = transpose(A, n, lda);
        zgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getrf_f(const int &m, const int &n, float *A, const int &lda, int *ipiv, int &info)
    {
        sgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    inline void getrf_f(const int &m, const int &n, double *A, const int &lda, int *ipiv, int &info)
    {
        dgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    inline void getrf_f(const int &m, const int &n, std::complex<float> *A, const int &lda, int *ipiv, int &info)
    {
        cgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    inline void getrf_f(const int &m, const int &n, std::complex<double> *A, const int &lda, int *ipiv, int &info)
    {
        zgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    inline void getri(const int &n, float *A, const int &lda, int *ipiv, float *work, const int &lwork, int &info)
    {
        float *a_fort = transpose(A, n, lda);
        sgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getri(const int &n, double *A, const int &lda, int *ipiv, double *work, const int &lwork, int &info)
    {
        double *a_fort = transpose(A, n, lda);
        dgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getri(const int &n, std::complex<float> *A, const int &lda, int *ipiv, std::complex<float> *work, const int &lwork, int &info)
    {
        std::complex<float> *a_fort = transpose(A, n, lda);
        cgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getri(const int &n, std::complex<double> *A, const int &lda, int *ipiv, std::complex<double> *work, const int &lwork, int &info)
    {
        std::complex<double> *a_fort = transpose(A, n, lda);
        zgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    inline void getri_f(const int &n, float *A, const int &lda, int *ipiv, float *work, const int &lwork, int &info)
    {
        sgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

    inline void getri_f(const int &n, double *A, const int &lda, int *ipiv, double *work, const int &lwork, int &info)
    {
        dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

    inline void getri_f(const int &n, std::complex<float> *A, const int &lda, int *ipiv, std::complex<float> *work, const int &lwork, int &info)
    {
        cgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

    inline void getri_f(const int &n, std::complex<double> *A, const int &lda, int *ipiv, std::complex<double> *work, const int &lwork, int &info)
    {
        zgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

    /* BLACS/ScaLAPACK wrappers */

    // suffix _f indicates the IO matrices stored in memory in fortran format, i.e. column-major
    inline
    void pgemm_f(const char &transa, const char &transb,
                 const int &M, const int &N, const int &K,
                 const float &alpha,
                 const float *A, const int &IA, const int &JA, const int *DESCA,
                 const float *B, const int &IB, const int &JB, const int *DESCB,
                 const float &beta,
                 float *C, const int &IC, const int &JC, const int *DESCC)
    {
        psgemm_(&transa, &transb, &M, &N, &K, &alpha,
                A, &IA, &JA, DESCA,
                B, &IB, &JB, DESCB,
                &beta,
                C, &IC, &JC, DESCC);
    }

    inline
    void pgemm_f(const char &transa, const char &transb,
                 const int &M, const int &N, const int &K,
                 const double &alpha,
                 const double *A, const int &IA, const int &JA, const int *DESCA,
                 const double *B, const int &IB, const int &JB, const int *DESCB,
                 const double &beta,
                 double *C, const int &IC, const int &JC, const int *DESCC)
    {
        pdgemm_(&transa, &transb, &M, &N, &K, &alpha,
                A, &IA, &JA, DESCA,
                B, &IB, &JB, DESCB,
                &beta,
                C, &IC, &JC, DESCC);
    }

    inline
    void pgemm_f(const char &transa, const char &transb,
                 const int &M, const int &N, const int &K,
                 const std::complex<float> &alpha,
                 const std::complex<float> *A, const int &IA, const int &JA, const int *DESCA,
                 const std::complex<float> *B, const int &IB, const int &JB, const int *DESCB,
                 const std::complex<float> &beta,
                 std::complex<float> *C, const int &IC, const int &JC, const int *DESCC)
    {
        pcgemm_(&transa, &transb, &M, &N, &K, &alpha,
                A, &IA, &JA, DESCA,
                B, &IB, &JB, DESCB,
                &beta,
                C, &IC, &JC, DESCC);
    }

    inline
    void pgemm_f(const char &transa, const char &transb,
                 const int &M, const int &N, const int &K,
                 const std::complex<double> &alpha,
                 const std::complex<double> *A, const int &IA, const int &JA, const int *DESCA,
                 const std::complex<double> *B, const int &IB, const int &JB, const int *DESCB,
                 const std::complex<double> &beta,
                 std::complex<double> *C, const int &IC, const int &JC, const int *DESCC)
    {
        pzgemm_(&transa, &transb, &M, &N, &K, &alpha,
                A, &IA, &JA, DESCA,
                B, &IB, &JB, DESCB,
                &beta,
                C, &IC, &JC, DESCC);
    }

    inline
    void blacs_gridinit(int &ictxt, const char order, const int nprow, const int npcol )
    {
        blacs_gridinit_(&ictxt, &order, &nprow, &npcol);
    }

    inline
    void blacs_gridinfo(const int &ictxt, int &nprow, int &npcol, int &myprow, int &mypcol )
    {
        blacs_gridinfo_(&ictxt, &nprow, &npcol, &myprow, &mypcol);
    }

    inline
    void blacs_pcoord(const int &ictxt, const int &pid, int &prow, int &pcol)
    {
        blacs_pcoord_(&ictxt, &pid, &prow, &pcol);
    }

    inline
    int blacs_pnum(const int &ictxt, const int &prow, const int &pcol)
    {
        return blacs_pnum_(&ictxt, &prow, &pcol);
    }

    inline
    int numroc(const int n, const int nb, const int iproc, const int srcproc, const int nprocs)
    {
        return numroc_(&n, &nb, &iproc, &srcproc, &nprocs);
    }

    inline
    void descinit(int *desc, 
                  const int m, const int n, const int mb, const int nb,
                  const int irsrc, const int icsrc, 
                  const int ictxt, const int lld, int &info)
    {
        descinit_(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);
    }

    inline
    void pgemr2d_f(const int m, const int n,
                   const float *a, const int ia, const int ja, const int *desca,
                   float *b, const int ib, const int jb, const int *descb,
                   const int ictxt)
    {
        psgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    inline
    void pgemr2d_f(const int m, const int n,
                   const double *a, const int ia, const int ja, const int *desca,
                   double *b, const int ib, const int jb, const int *descb,
                   const int ictxt)
    {
        pdgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    inline
    void pgemr2d_f(const int m, const int n,
                   const std::complex<float> *a, const int ia, const int ja, const int *desca,
                   std::complex<float> *b, const int ib, const int jb, const int *descb,
                   const int ictxt)
    {
        pcgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    inline
    void pgemr2d_f(const int m, const int n,
                   const std::complex<double> *a, const int ia, const int ja, const int *desca,
                   std::complex<double> *b, const int ib, const int jb, const int *descb,
                   const int ictxt)
    {
        pzgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }
} /* namespace linalg */
