#pragma once

#include <complex>
#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    void blacs_gridinit_(int *ictxt, const char *order, const int *nprow, const int *npcol);
    void blacs_gridinfo_(const int *ictxt, int *nprow, int *npcol, int *myprow, int *mypcol);
    int numroc_(const int *n, const int *nb, const int *iproc, const int *srcproc, const int *nprocs);
    void descinit_(int *desc,
                   const int *m, const int *n, const int *mb, const int *nb,
                   const int *irsrc, const int *icsrc, const int *ictxt, const int *lld,
                   int *info);

    void pdpotrf_(char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info);
    void pzpotrf_(char *uplo, int *n, std::complex<double> *a, int *ia, int *ja, int *desca, int *info);
    void pdtran_(int *m, int *n, double *alpha,
                 double *a, int *ia, int *ja, int *desca,
                 double *beta, 
                 double *c, int *ic, int *jc, int *descc);
    // matrix-vector
    void pzgemv_(const char *transa, const int *M, const int *N, const std::complex<double> *alpha,
                 const std::complex<double> *A, const int *IA, const int *JA, const int *DESCA,
                 const std::complex<double> *B, const int *IB, const int *JB, const int *DESCB,
                 const int *K, const std::complex<double> *beta,
                 std::complex<double> *C, const int *IC, const int *JC, const int *DESCC,
                 const int *L);
    void pdgemv_(const char *transa, const int *M, const int *N, const double *alpha,
                 const double *A, const int *IA, const int *JA, const int *DESCA,
                 const double *B, const int *IB, const int *JB, const int *DESCB,
                 const int *K, const double *beta,
                 double *C, const int *IC, const int *JC, const int *DESCC,
                 const int *L);
    // matrix-matrix
    void psgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const float *alpha,
                 const float *A, const int *IA, const int *JA, const int *DESCA,
                 const float *B, const int *IB, const int *JB, const int *DESCB,
                 const float *beta,
                 float *C, const int *IC, const int *JC, const int *DESCC);
    void pcgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const std::complex<float> *alpha,
                 const std::complex<float> *A, const int *IA, const int *JA, const int *DESCA,
                 const std::complex<float> *B, const int *IB, const int *JB, const int *DESCB,
                 const std::complex<float> *beta,
                 std::complex<float> *C, const int *IC, const int *JC, const int *DESCC);
    void pdgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const double *alpha,
                 const double *A, const int *IA, const int *JA, const int *DESCA,
                 const double *B, const int *IB, const int *JB, const int *DESCB,
                 const double *beta,
                 double *C, const int *IC, const int *JC, const int *DESCC);
    void pzgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const std::complex<double> *alpha,
                 const std::complex<double> *A, const int *IA, const int *JA, const int *DESCA,
                 const std::complex<double> *B, const int *IB, const int *JB, const int *DESCB,
                 const std::complex<double> *beta,
                 std::complex<double> *C, const int *IC, const int *JC, const int *DESCC);
    void pdsymm_(const char *side, const char *uplo,
                 const int *m, const int *n, const double *alpha,
                 const double *a, const int *ia, const int *ja, const int *desca,
                 const double *b, const int *ib, const int *jb, const int *descb,
                 const double *beta,
                 double *c, const int *ic, const int *jc, const int *descc);
    void pdtrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                 const int *m, const int *n,
                 const double *alpha,
                 const double *a, const int *ia, const int *ja, const int *desca,
                 double *b, const int *ib, const int *jb, const int *descb);
    void pztrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                 const int *m, const int *n,
                 const double *alpha, const std::complex<double> *a, const int *ia, const int *ja, const int *desca,
                 std::complex<double> *b, const int *ib, const int *jb, const int *descb);

    void pzgetrf_(const int *M, const int *N, 
                  std::complex<double> *A, const int *IA, const int *JA, const int *DESCA,
                  int *ipiv, int *info);

    void pdsygvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                  const int *n, double *A, const int *ia, const int *ja, const int*desca,
                  double *B, const int *ib, const int *jb, const int*descb,
                  const double *vl, const double *vu, const int *il, const int *iu,
                  const double *abstol, int *m, int *nz, double *w, const double *orfac,
                  double *Z, const int *iz, const int *jz, const int *descz,
                  double *work, int *lwork, int *iwork, int *liwork, int *ifail, int *iclustr,
                  double *gap, int *info);
    void pzhegvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                  const int *n, std::complex<double> *A, const int *ia, const int *ja, const int *desca,
                  std::complex<double> *B, const int *ib, const int *jb, const int *descb,
                  const double *vl, const double *vu, const int *il, const int *iu,
                  const double *abstol, int *m, int *nz, double *w, const double *orfac,
                  std::complex<double> *Z, const int *iz, const int *jz, const int*descz,
                  std::complex<double> *work, int *lwork, double *rwork, int *lrwork, int *iwork, int*liwork, int *ifail, int*iclustr,
                  double*gap, int *info);

    void pzgetri_(const int *n, 
                  std::complex<double> *A, const int *ia, const int *ja, const int *desca,
                  int *ipiv, const std::complex<double> *work, const int *lwork, const int *iwork, const int *liwork,
                  const int *info);

    void pzgeadd_(const char *transa, const int *m, const int *n,
                  const std::complex<double> *alpha,
                  const std::complex<double> *a, const int *ia, const int *ja, const int *desca,
                  const std::complex<double> *beta,
                  std::complex<double> *c, const int *ic, const int *jc, const int *descc);
#ifdef __cplusplus
}
#endif /* __cplusplus */
