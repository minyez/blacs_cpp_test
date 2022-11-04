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
    matrix<double> mat1(1, 1, MAJOR::COL), mat2(1, 1, MAJOR::COL), prod(1, 1, MAJOR::COL);
    // auto hemat = random_he(5, complex<double>{0, 0}, complex<double>{1, 0});
    // auto hemat_inv = inverse(hemat);
    // printf("hemat*hemat_inv\n%s", str(hemat*hemat_inv).c_str());

    if (blacs_h.myid == 0)
    {
        mat1.resize(m, k);
        mat2.resize(k, n);
        prod.resize(m, n);
        mat1.random(0, 2);
        mat2.random(-1, 1);
        printf("Full mat1\n%s", str(mat1).c_str());
        // printf("mathe\n%s", str(hemat).c_str());
        printf("Full mat2\n%s", str(mat2).c_str());
        auto prod_direct = mat1 * mat2;
        printf("mat1*mat2\n%s", str(prod_direct).c_str());
    }
    blacs_h.barrier();

    // array descriptor for storage of distributed data
    ArrayDesc mat1_desc(blacs_h), mat2_desc(blacs_h), prod_desc(blacs_h);
    // array descriptor. Full block descriptor to store global data only in the first process
    ArrayDesc mat1_block_desc(blacs_h), mat2_block_desc(blacs_h), prod_block_desc(blacs_h);
    mat1_desc.init(m, k, mb, kb, IRSRC, ICSRC);
    mat1_block_desc.init(m, k, m, k, IRSRC, ICSRC);
    mat2_desc.init(k, n, kb, nb, IRSRC, ICSRC);
    mat2_block_desc.init(k, n, k, n, IRSRC, ICSRC);
    prod_desc.init(m, n, mb, nb, IRSRC, ICSRC);
    prod_block_desc.init(m, n, m, n, IRSRC, ICSRC);

    matrix<double> mat1_local = init_local_mat<decltype(mat1)::type>(mat1_desc, MAJOR::COL);
    matrix<double> mat2_local = init_local_mat<decltype(mat1)::type>(mat2_desc, MAJOR::COL);
    matrix<double> prod_local = init_local_mat<decltype(mat1)::type>(prod_desc, MAJOR::COL);

    // distribute data to other processes
    linalg::pgemr2d_f(m, k, mat1.c, 1, 1, mat1_block_desc.desc,
                      mat1_local.c, 1, 1, mat1_desc.desc, blacs_h.ictxt);
    linalg::pgemr2d_f(k, n, mat2.c, 1, 1, mat2_block_desc.desc,
                      mat2_local.c, 1, 1, mat2_desc.desc, blacs_h.ictxt);

    linalg::pgemm_f('N', 'N', m, n, k, 1.0, mat1_local.c, 1, 1, mat1_desc.desc,
                    mat2_local.c, 1, 1, mat2_desc.desc, 0.0, prod_local.c, 1, 1, prod_desc.desc);
    printf("prod_local on %s\n%s", prod_desc.info().c_str(), str(prod_local).c_str());
    // gather 
    linalg::pgemr2d_f(m, n, prod_local.c, 1, 1, prod_desc.desc,
                      prod.c, 1, 1, prod_block_desc.desc, blacs_h.ictxt);

    if (blacs_h.myid == 0)
        printf("prod gathered on %s\n%s", prod_desc.info().c_str(), str(prod).c_str());

    blacs_h.barrier();
    mpi.finalize();
    return 0;
}
