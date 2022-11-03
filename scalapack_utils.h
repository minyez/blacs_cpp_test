#pragma once
#include "mpi_utils.h"
#include <utility>
#include <vector>
#include <string>

std::vector<std::pair<int, int>> get_all_pcoords(int nprows, int npcols);
std::pair<int, int> get_pcoord_from_pid(int pid, int nprows, int npcols, char layout);
int get_pid_from_pcoord(int nprows, int prow, int npcols, int pcol, char layout);
int global_index(int lindex, int n, int nb, int nprc, int myprc);
int local_index(int gindex, int n, int nb, int nprc, int myprc);

class BLACS_handler: public MPI_handler
{
private:
    void init();
public:
    int nprows;
    int npcols;
    int ictxt;
    char layout;
    int myprow;
    int mypcol;
    BLACS_handler(int comm_in, const char &layout_in = 'R', bool greater_nprows = true);
    BLACS_handler(int comm_in, const char &layout_in, const int &nprows_in, const int &npcols_in);
    ~BLACS_handler() {};
    std::string info() const;
    //! obtain the process coordinate by process ID in the current context and grid layout
    std::pair<int, int> get_pcoord(int pid);
};

class ArrayDesc
{
private:
    // BLACS parameters obtained upon construction
    int ictxt_;
    int nprocs_;
    int myid_;
    int nprows_;
    int myprow_;
    int npcols_;
    int mypcol_;
    void set_blacs_params_(int ictxt, int nprocs, int myid, int nprows, int myprow, int npcols, int mypcol);

    // Array dimensions
    int m_;
    int n_;
    int mb_;
    int nb_;
    int irsrc_;
    int icsrc_;
    int lld_;
    int m_local_;
    int n_local_;

    //! flag to indicate that the current process should contain no data of local matrix, but for scalapack routines, it will generate a dummy matrix of size 1, nrows = ncols = 1
    bool gen_dummy_matrix_ = false;

    //! flag for initialization
    bool initialized_ = false;
public:
    int desc[9];
    ArrayDesc(const BLACS_handler &blacs_h);
    ArrayDesc(const BLACS_handler &blacs_h, int pid);
    ArrayDesc(const BLACS_handler &blacs_h, int prow, int pcol);
    ArrayDesc(const int &ictxt);
    ArrayDesc(const int &ictxt, int pid);
    ArrayDesc(const int &ictxt, int prow, int pcol);
    int init(const int &m, const int &n,
             const int &mb, const int &nb,
             const int &irsrc, const int &icsrc);
    int to_loid_r(int goid) const;
    int to_loid_c(int goid) const;
    int to_goid_r(int loid) const;
    int to_goid_c(int loid) const;
    int m() const { return m_; }
    int n() const { return n_; }
    int mb() const { return mb_; }
    int nb() const { return nb_; }
    int irsrc() const { return irsrc_; }
    int icsrc() const { return icsrc_; }
    int num_r() const { return m_local_; }
    int num_c() const { return n_local_; }
    std::string info() const;
};
