#pragma once
#include "mpi_utils.h"
#include <string>

int global_index(int lindex, int n, int nb, int nprc, int myprc);
int local_index(int gindex, int n, int nb, int nprc, int myprc);

class BLACS_handler: public MPI_handler
{
private:
    void set_blacs_params();
public:
    int nprows;
    int npcols;
    int ictxt;
    char layout;
    int myprow;
    int mypcol;
    BLACS_handler(int comm_in);
    BLACS_handler(int comm_in, const char &layout_in, const int &nprows_in, const int &npcols_in);
    ~BLACS_handler() {};
    std::string info() const;
};

class ArrayDesc
{
private:
    const BLACS_handler blacs_h_;
    int m_;
    int n_;
    int mb_;
    int nb_;
    int irsrc_;
    int icsrc_;
    int ldd_;
    int m_local_;
    int n_local_;
public:
    int desc[9];
    ArrayDesc(const BLACS_handler &blacs_h): blacs_h_(blacs_h) {};
    int init(const int &m, const int &n,
             const int &mb, const int &nb,
             const int &irsrc, const int &icsrc,
             int ldd = -1);
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
