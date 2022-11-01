#pragma once

class MPI_wrapper
{
public:
    MPI_wrapper() {}
    void init(int *argc, char ***argv);
    void init(int *argc, char ***argv, int thread_option);
    void init_thread(int mpi_thread);
    void finalize();
    unsigned long comm_world;
};

class MPI_handler
{
public:
    int comm;
    int nprocs;
    int myid;
    MPI_handler(int comm_);
    void barrier();
};
