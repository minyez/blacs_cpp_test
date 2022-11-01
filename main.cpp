#include "main.h"
#include "linalg.h"
#include <iostream>
#ifdef __MPI__
#include <mpi.h>
#endif

using namespace std;

int main (int argc, char *argv[])
{
    int provided, info;
    MPI_wrapper mpi;
    mpi.init(&argc, &argv);

    BLACS_handler blacs_h(mpi.comm_world);
    printf("%s\n", blacs_h.info().c_str());
    blacs_h.barrier();

    const int m = 3, n = 4, k = 2;
    const int mb = 1, nb= 1, kb = 1;
    const int IRSRC = 0, ICSRC = 0;

    matrix<double> mat1(m, k);
    matrix<double> mat2(k, n);
    matrix<double> mat3(m, n);
    // auto hemat = random_he(5, complex<double>{0, 0}, complex<double>{1, 0});
    // auto hemat_inv = inverse(hemat);
    // printf("hemat*hemat_inv\n%s", str(hemat*hemat_inv).c_str());

    mat1.random(0, 2);
    mat2.random(-1, 1);
    assert(mat1 != mat2);

    auto prod_direct = mat1 * mat2;
    if (blacs_h.myid == 0)
    {
        printf("mat1\n%s", str(mat1).c_str());
        // printf("mathe\n%s", str(hemat).c_str());
        printf("mat2\n%s", str(mat2).c_str());
        printf("mat1*mat2\n%s", str(prod_direct).c_str());
    }
    blacs_h.barrier();

    ArrayDesc desc_mat1(blacs_h), desc_mat2(blacs_h), desc_mat3(blacs_h);
    desc_mat1.init(m, k, mb, kb, IRSRC, ICSRC);
    desc_mat2.init(k, n, kb, nb, IRSRC, ICSRC);
    desc_mat3.init(m, n, mb, nb, IRSRC, ICSRC);

    // initialize local matrices with column-major memory arrangement
    auto mat1_local = get_local_mat(mat1, desc_mat1, MAJOR::COL);
    auto mat2_local = get_local_mat(mat2, desc_mat2, MAJOR::COL);
    auto mat3_local = get_local_mat(mat3, desc_mat3, MAJOR::COL);
    assert(mat1_local.is_col_major() && mat2_local.is_col_major() && mat3_local.is_col_major());

    printf("mat1_local of %s\n%s", desc_mat1.info().c_str(), str(mat1_local).c_str());
    printf("mat2_local of %s\n%s", desc_mat2.info().c_str(), str(mat2_local).c_str());
    const char transn = 'N';
    double alpha = 1.0, beta = 0.0;
    int i1 = 1;
    int desc1[9], desc2[9], desc3[9];
    // for row-major matrices
    // linalg::descinit(desc2, n, k, nb, mb, IRSRC, ICSRC, blacs_h.ictxt, mat2_local.nc(), info);
    // linalg::descinit(desc1, k, m, kb, mb, IRSRC, ICSRC, blacs_h.ictxt, mat1_local.nc(), info);
    // linalg::descinit(desc3, n, m, nb, mb, IRSRC, ICSRC, blacs_h.ictxt, mat3_local.nc(), info);
    // pdgemm_(&transn, &transn, &n, &m, &k, &alpha, mat2_local.c, &i1, &i1, desc2,
    //         mat1_local.c, &i1, &i1, desc1, &beta, mat3_local.c, &i1, &i1, desc3);

    // for col-major matrices
    linalg::descinit(desc1, m, k, mb, kb, IRSRC, ICSRC, blacs_h.ictxt, mat1_local.nr(), info);
    linalg::descinit(desc2, k, n, kb, nb, IRSRC, ICSRC, blacs_h.ictxt, mat2_local.nr(), info);
    linalg::descinit(desc3, m, n, mb, nb, IRSRC, ICSRC, blacs_h.ictxt, mat3_local.nr(), info);
    linalg::pgemm_f('N', 'N', m, n, k, alpha, mat1_local.c, 1, 1, desc1,
                    mat2_local.c, 1, 1, desc2, beta, mat3_local.c, 1, 1, desc3);

    blacs_h.barrier();
    printf("mat3_local of %s\n%s", desc_mat3.info().c_str(), str(mat3_local).c_str());

    mpi.finalize();
    return 0;
}
