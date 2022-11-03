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

    BLACS_handler blacs_h(mpi.comm_world, 'R', false);
    printf("%s\n", blacs_h.info().c_str());
    blacs_h.barrier();

    const int m = 3, n = 4, k = 2;
    const int mb = 1, nb= 1, kb = 1;
    const int IRSRC = 0, ICSRC = 0;

    // define the matrices, with no data, i.e. no memory allocated
    matrix<double> mat1, mat2;
    // auto hemat = random_he(5, complex<double>{0, 0}, complex<double>{1, 0});
    // auto hemat_inv = inverse(hemat);
    // printf("hemat*hemat_inv\n%s", str(hemat*hemat_inv).c_str());


    mat1.resize(m, k);
    mat2.resize(k, n);
    mat1.random(0, 2);
    mat2.random(-1, 1);

    if (blacs_h.myid == 0)
    {
        printf("Full mat1\n%s", str(mat1).c_str());
        // printf("mathe\n%s", str(hemat).c_str());
        printf("Full mat2\n%s", str(mat2).c_str());
        auto prod_direct = mat1 * mat2;
        printf("mat1*mat2\n%s", str(prod_direct).c_str());
    }
    blacs_h.barrier();

    // array descriptor of own process
    ArrayDesc mat1_desc(blacs_h), mat2_desc(blacs_h), prod_desc(blacs_h);
    mat1_desc.init(m, k, mb, kb, IRSRC, ICSRC);
    mat2_desc.init(k, n, kb, nb, IRSRC, ICSRC);
    prod_desc.init(m, n, mb, nb, IRSRC, ICSRC);
    // matrix<double> mat1_local = init_local_mat<decltype(mat1)::type>(mat1_desc, MAJOR::COL);
    // matrix<double> mat2_local = init_local_mat<decltype(mat1)::type>(mat2_desc, MAJOR::COL);
    matrix<double> prod_local = init_local_mat<decltype(mat1)::type>(prod_desc, MAJOR::COL);

    // initialize local matrices with column-major memory arrangement
    auto mat1_local = get_local_mat(mat1, mat1_desc, MAJOR::COL);
    auto mat2_local = get_local_mat(mat2, mat2_desc, MAJOR::COL);
    // auto mat3_local = init_local_mat<decltype(mat1_local)::type>(mat3_desc, MAJOR::COL);

    printf("mat1_local of %s\n%s", mat1_desc.info().c_str(), str(mat1_local).c_str());
    printf("mat2_local of %s\n%s", mat2_desc.info().c_str(), str(mat2_local).c_str());
    // for (int pid = 1; pid != blacs_h.nprocs; pid++)
    // {
    //     if (blacs_h.myid == 0)
    //     {
    //         // extract matrix from master process
    //         auto mat1_local = get_local_mat_pid(mat1, mb, kb, IRSRC, ICSRC, blacs_h, pid, MAJOR::COL);
    //         printf("mat1_local extract from PID 0 for PID %d\n%s", pid, str(mat1_local).c_str());
    //         // send to the corresponding pid
    //     }
    //     else
    //     {
    //
    //     }
    // }
    // const char transn = 'N';
    // double alpha = 1.0, beta = 0.0;
    // int i1 = 1;
    // int desc1[9], desc2[9], desc3[9];
    // for row-major matrices
    // linalg::descinit(desc2, n, k, nb, mb, IRSRC, ICSRC, blacs_h.ictxt, mat2_local.nc(), info);
    // linalg::descinit(desc1, k, m, kb, mb, IRSRC, ICSRC, blacs_h.ictxt, mat1_local.nc(), info);
    // linalg::descinit(desc3, n, m, nb, mb, IRSRC, ICSRC, blacs_h.ictxt, mat3_local.nc(), info);
    // pdgemm_(&transn, &transn, &n, &m, &k, &alpha, mat2_local.c, &i1, &i1, desc2,
    //         mat1_local.c, &i1, &i1, desc1, &beta, mat3_local.c, &i1, &i1, desc3);

    linalg::pgemm_f('N', 'N', m, n, k, 1.0, mat1_local.c, 1, 1, mat1_desc.desc,
                    mat2_local.c, 1, 1, mat2_desc.desc, 0.0, prod_local.c, 1, 1, prod_desc.desc);

    printf("prod_local of %s\n%s", prod_desc.info().c_str(), str(prod_local).c_str());

    blacs_h.barrier();
    mpi.finalize();
    return 0;
}
