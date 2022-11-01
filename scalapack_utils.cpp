#include "scalapack_utils.h"
#include "linalg.h"
#include <stdexcept>
#include <cmath>
#ifdef __MPI__
#include <mpi.h>
#endif

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

BLACS_handler::BLACS_handler(int comm_in): MPI_handler(comm_in)
{
    layout = 'C';
    int nc = std::ceil(std::sqrt(double(nprocs)));
    for (; nc != 1; nc--)
    {
        if((nprocs)%nc==0) break;
    }
    npcols = nc;
    nprows = nprocs / npcols;
    set_blacs_params();
}

BLACS_handler::BLACS_handler(int comm_in, const char &layout_in, const int &nprows_in, const int &npcols_in): MPI_handler(comm_in)
{
    if (nprocs != nprows_in * npcols_in)
        throw std::invalid_argument("nprocs != nprows * npcols");
    nprows = nprows_in;
    npcols = npcols_in;
    layout = layout_in;
    set_blacs_params();
}

void BLACS_handler::set_blacs_params()
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

std::string BLACS_handler::info() const
{
    std::string info;
    info = std::string("BLACS handler: ")
         + "ICTXT " + std::to_string(ictxt) + " "
         + "PSIZE " + std::to_string(nprocs) + " "
         + "PID " + std::to_string(myid) + " "
         + "PGRID (" + std::to_string(nprows) + "," + std::to_string(npcols) + ") "
         + "PCORD (" + std::to_string(myprow) + "," + std::to_string(mypcol) +")";
    return info;
}

int ArrayDesc::init(const int &m, const int &n, const int &mb, const int &nb, const int &irsrc, const int &icsrc)
{
    int info;
#ifdef __MPI__
    m_local_ = linalg::numroc(m, mb, blacs_h_.myprow, irsrc, blacs_h_.nprows);
    n_local_ = linalg::numroc(n, nb, blacs_h_.mypcol, icsrc, blacs_h_.npcols);
    linalg::descinit(desc, m, n, mb, nb, irsrc, icsrc, blacs_h_.ictxt, m_local_, info);
    m_ = desc[2];
    n_ = desc[3];
    mb_ = desc[4];
    nb_ = desc[5];
    irsrc_ = desc[6];
    icsrc_ = desc[7];
    lld_ = desc[8];
#else
    desc[0] = 1;
    desc[1] = blacs_h_.ictxt;
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
    return info;
}

int ArrayDesc::to_loid_r(int goid) const
{
    return local_index(goid, m_, mb_, blacs_h_.nprows, blacs_h_.myprow);
}

int ArrayDesc::to_loid_c(int goid) const
{
    return local_index(goid, n_, nb_, blacs_h_.npcols, blacs_h_.mypcol);
}

int ArrayDesc::to_goid_r(int loid) const
{
    return global_index(loid, m_, mb_, blacs_h_.nprows, blacs_h_.myprow);
}

int ArrayDesc::to_goid_c(int loid) const
{
    return global_index(loid, n_, nb_, blacs_h_.npcols, blacs_h_.mypcol);
}

std::string ArrayDesc::info() const
{
    std::string info;
    info = std::string("ArrayDesc: ")
         + "ICTXT " + std::to_string(blacs_h_.ictxt) + " "
         + "ID " + std::to_string(blacs_h_.myid) + " "
         + "PCORD (" + std::to_string(blacs_h_.myprow) + "," + std::to_string(blacs_h_.mypcol) + ") "
         + "GSIZE (" + std::to_string(m_) + "," + std::to_string(n_) + ") "
         + "LSIZE (" + std::to_string(m_local_) + "," + std::to_string(n_local_) + ")";
    return info;
}
