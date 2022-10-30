#include "mpi_utils.h"
#ifdef __MPI__
#include <mpi.h>
#endif

void MPI_wrapper::init(int *argc, char ***argv)
{
#ifdef __MPI__
    MPI_Init(argc, argv);
    comm_world = MPI_COMM_WORLD;
#else
    comm_world = 0;
#endif
}

void MPI_wrapper::finalize()
{
#ifdef __MPI__
    MPI_Finalize();
#endif
}

MPI_handler::MPI_handler(int comm_): comm(comm_)
{
#ifdef __MPI__
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &nprocs);
#else
    nprocs = 1;
    myid = 0;
#endif
}

void MPI_handler::barrier()
{
#ifdef __MPI__
    MPI_Barrier(comm);
#endif
}
