#include "scalapack_utils.h"
#include "linalg.h"
#include <cassert>
#include <stdexcept>
#include <cmath>
#ifdef __MPI__
#include <mpi.h>
#endif

std::pair<int, int> get_pcoord_from_pid(int pid, int nprows, int npcols, char layout)
{
    if (layout == 'R' || layout == 'r')
        return std::pair<int, int>{pid / npcols, pid % npcols};
    if (layout == 'C' || layout == 'c')
        return std::pair<int, int>{pid / nprows, pid % nprows};
    throw std::invalid_argument("Known layout");
}

std::vector<std::pair<int, int>> get_all_pcoords(int nprows, int npcols)
{
    std::vector<std::pair<int, int>> pcoords;
    for (int i = 0; i != nprows; i++)
        for (int j = 0; j != npcols; j++)
            pcoords.push_back({i, j});
    return pcoords;
}

int get_pid_from_pcoord(int nprows, int prow, int npcols, int pcol, char layout)
{
    if (layout == 'R' || layout == 'r')
        return npcols * prow + pcol;
    if (layout == 'C' || layout == 'c')
        return nprows * pcol + prow;
    throw std::invalid_argument("Known layout");
}

int global_index(int lindex, int n, int nb, int nprc, int myprc)
{
    int iblock, gi;
    iblock = lindex / nb;
    gi = (iblock * nprc + myprc)*nb + lindex % nb;
    return gi;
}

int local_index(int gindex, int n, int nb, int nprc, int myprc)
{
    int inproc = int((gindex % (nb*nprc)) / nb);
    return myprc == inproc? int(gindex / (nb*nprc))*nb + gindex % nb : -1;
}

int optimal_blocksize(int n, int nprc)
{
    // the optimal blocksize 
    int nb = 1;
    return nb;
}

BLACS_handler::BLACS_handler(int comm_in, const char &layout_in, bool greater_nprows): MPI_handler(comm_in)
{
    layout = layout_in;
    // 'C' layout may lead to inconsistency
    int nsqrt = std::floor(std::sqrt(double(nprocs)));
    for (; nsqrt != 1; nsqrt--)
    {
        if((nprocs)%nsqrt==0) break;
    }
    if (greater_nprows)
        nprows = nprocs / (npcols = nsqrt);
    else
        npcols = nprocs / (nprows = nsqrt);
    init();
}

BLACS_handler::BLACS_handler(int comm_in, const char &layout_in, const int &nprows_in, const int &npcols_in): MPI_handler(comm_in)
{
    if (nprocs != nprows_in * npcols_in)
        throw std::invalid_argument("nprocs != nprows * npcols");
    nprows = nprows_in;
    npcols = npcols_in;
    layout = layout_in;
    init();
}

void BLACS_handler::init()
{
#ifdef __MPI__
    // ictxt = MPI_Comm_c2f(comm);
    // blacs_gridinit_(&ictxt, &layout, &nprows, &npcols);
    // blacs_gridinfo_(&ictxt, &nprows, &npcols, &myprow, &mypcol);
    ictxt = Csys2blacs_handle(comm);
    Cblacs_gridinit(&ictxt, &layout, nprows, npcols);
    Cblacs_gridinfo(ictxt, &nprows, &npcols, &myprow, &mypcol);
#else
    ictxt = comm;
    myprow = mypcol = 0;
#endif
}

std::pair<int, int> BLACS_handler::get_pcoord(int pid)
{
    int prow = 0, pcol = 0;
#ifdef __MPI__
    blacs_pcoord_(&(this->ictxt), &pid, &prow, &pcol);
#endif
    return std::pair<int, int>{prow, pcol};
}

std::string BLACS_handler::info() const
{
    std::string info;
    info = std::string("BLACS handler: ")
         + "ICTXT " + std::to_string(ictxt) + " "
         + "PSIZE " + std::to_string(nprocs) + " "
         + "PID " + std::to_string(myid) + " "
         + "PGRID (" + std::to_string(nprows) + "," + std::to_string(npcols) + ") "
         + "PCOOD (" + std::to_string(myprow) + "," + std::to_string(mypcol) +")";
    return info;
}

void ArrayDesc::set_blacs_params_(int ictxt, int nprocs, int myid, int nprows, int myprow, int npcols, int mypcol)
{
    assert(myid < nprocs && myprow < nprows && mypcol < npcols);
    ictxt_ = ictxt;
    nprocs_ = nprocs;
    myid_ = myid;
    nprows_ = nprows;
    myprow_ = myprow;
    npcols_ = npcols;
    mypcol_ = mypcol;
}

ArrayDesc::ArrayDesc(const BLACS_handler &blacs_h)
{
    set_blacs_params_(blacs_h.ictxt, blacs_h.nprocs, blacs_h.myid,
                      blacs_h.nprows, blacs_h.myprow,
                      blacs_h.npcols, blacs_h.mypcol);
}

ArrayDesc::ArrayDesc(const BLACS_handler &blacs_h, int pid)
{
    int prow, pcol;
#ifdef __MPI__
    linalg::blacs_pcoord(blacs_h.ictxt, pid, prow, pcol);
#else
    prow = 0;
    pcol = 0;
#endif
    set_blacs_params_(blacs_h.ictxt, blacs_h.nprocs, pid,
                      blacs_h.nprows, prow,
                      blacs_h.npcols, pcol);
}

ArrayDesc::ArrayDesc(const BLACS_handler &blacs_h, int prow, int pcol)
{
    int pid;
#ifdef __MPI__
    pid = linalg::blacs_pnum(blacs_h.ictxt, prow, pcol);
#else
    pid = 0;
#endif
    set_blacs_params_(blacs_h.ictxt, blacs_h.nprocs, pid,
                      blacs_h.nprows, prow,
                      blacs_h.npcols, pcol);
}

int ArrayDesc::init(const int &m, const int &n, const int &mb, const int &nb, const int &irsrc, const int &icsrc)
{
    int info;
#ifdef __MPI__
    m_local_ = linalg::numroc(m, mb, myprow_, irsrc, nprows_);
    // leading dimension is always max(1, numroc)
    lld_ = std::max(m_local_, 1);
    n_local_ = linalg::numroc(n, nb, mypcol_, icsrc, npcols_);
    // create a dummy matrix of size 1, such that pointer c is not nullptr
    // this is VERY IMPORTANT when calling scalapack with small matrix for many processors
    if (m_local_ < 1 || n_local_ < 1)
    {
        gen_dummy_matrix_ = true;
        m_local_ = 1;
        n_local_ = 1;
    }

    linalg::descinit(this->desc, m, n, mb, nb, irsrc, icsrc, ictxt_, lld_, info);
    if (info)
        printf("ERROR DESCINIT! PROC %d (%d,%d) PARAMS: DESC %d %d %d %d %d %d %d %d\n", myid_, myprow_, mypcol_, m, n, mb, nb, irsrc, icsrc, ictxt_, m_local_);
    // else
    //     printf("SUCCE DESCINIT! PROC %d (%d,%d) PARAMS: DESC %d %d %d %d %d %d %d %d\n", myid_, myprow_, mypcol_, m, n, mb, nb, irsrc, icsrc, ictxt_, m_local_);
    m_ = desc[2];
    n_ = desc[3];
    mb_ = desc[4];
    nb_ = desc[5];
    irsrc_ = desc[6];
    icsrc_ = desc[7];
    lld_ = desc[8];
#else
    desc[0] = 1;
    desc[1] = ictxt_;
    desc[2] = m_ = m;
    desc[3] = n_ = n;
    desc[4] = mb_ = mb;
    desc[5] = nb_ = nb;
    desc[6] = irsrc_ = irsrc;
    desc[7] = icsrc_ = icsrc;
    desc[8] = lld_ = m;
    m_local_ = m_;
    n_local_ = n_;
    lld_ = m_local_;
    info = 0;
#endif
    initialized_ = true;
    return info;
}

int ArrayDesc::to_loid_r(int goid) const
{
    return local_index(goid, m_, mb_, nprows_, myprow_);
}

int ArrayDesc::to_loid_c(int goid) const
{
    return local_index(goid, n_, nb_, npcols_, mypcol_);
}

int ArrayDesc::to_goid_r(int loid) const
{
    return global_index(loid, m_, mb_, nprows_, myprow_);
}

int ArrayDesc::to_goid_c(int loid) const
{
    return global_index(loid, n_, nb_, npcols_, mypcol_);
}

std::string ArrayDesc::info() const
{
    std::string info;
    info = std::string("ArrayDesc: ")
         + "ICTXT " + std::to_string(ictxt_) + " "
         + "ID " + std::to_string(myid_) + " "
         + "PCOOR (" + std::to_string(myprow_) + "," + std::to_string(mypcol_) + ") "
         + "GSIZE (" + std::to_string(m_) + "," + std::to_string(n_) + ") "
         + "LSIZE (" + std::to_string(m_local_) + "," + std::to_string(n_local_) + ") "
         + "DUMMY? " + std::string(gen_dummy_matrix_? "T" : "F");
    return info;
}
