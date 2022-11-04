#pragma once
#include <cassert>
#include <vector>
// #include <iostream>
#include <cmath>
#include <ctime>
#include <random>
#include "base.h"
#include "linalg.h"
#include "mpi_utils.h"
#include "scalapack_utils.h"
#include "vec.h"

// major dependent flattend index of 2D matrix
// row-major
inline int flatid_rm(const int &nr, const int &ir, const int &nc, const int &ic) { return ir*nc+ic; }
// column-major
inline int flatid_cm(const int &nr, const int &ir, const int &nc, const int &ic) { return ic*nr+ir; }
// wrapper
inline int flatid(const int &nr, const int &ir, const int &nc, const int &ic, bool row_major)
{
    return row_major? flatid_rm(nr, ir, nc, ic) : flatid_cm(nr, ir, nc, ic);
}

enum MAJOR { ROW, COL };

template <typename T>
class matrix
{
private:
    MAJOR major_;
    int mrank_;
    int size_;
    int nr_;
    int nc_;
public:
    using type = T;
    using real_t = typename to_real<T>::type;
    using cplx_t = typename to_cplx<T>::type;

    constexpr static const double EQUAL_THRES = DOUBLE_EQUAL_THRES;
    bool is_complex() { return is_complex_t<T>::value; }
    T *c;
    matrix() : nr_(0), nc_(0), major_(MAJOR::ROW), c(nullptr) { mrank_ = size_ = 0; }
    matrix(const int &nrows, const int &ncols, MAJOR major = MAJOR::ROW): nr_(nrows), nc_(ncols), major_(major), c(nullptr)
    {
        if (nr_&&nc_)
        {
            c = new T [nr_*nc_];
            mrank_ = std::min(nr_, nc_);
            size_ = nr_ * nc_;
        }
        zero_out();
    }
    matrix(const std::vector<vec<T>> &nested_vec, MAJOR major = MAJOR::ROW): major_(major), c(nullptr)
    {
        get_nr_nc_from_nested_vec(nested_vec, nr_, nc_);
        if (nr_&&nc_)
        {
            c = new T [nr_*nc_];
            mrank_ = std::min(nr_, nc_);
            size_ = nr_ * nc_;
            expand_nested_vector_to_pointer(nested_vec, nr_, nc_, c, is_row_major());
        }
    }

    matrix(const int &nrows, const int &ncols, const T * const valarr, MAJOR major = MAJOR::ROW): nr_(nrows), nc_(ncols), major_(major), c(nullptr)
    {
        if (nr_&&nc_)
        {
            mrank_ = std::min(nr_, nc_);
            size_ = nr_ * nc_;
            c = new T [nr_*nc_];
        }
        zero_out();
        // do not manually check out of bound
        // data are copied from valarr as it is, despite the major
        for (int i = 0; i < size(); i++)
            c[i] = valarr[i];
    }

    matrix(const matrix &m): nr_(m.nr_), nc_(m.nc_), major_(m.major_), c(nullptr)
    {
        if( nr_ && nc_ )
        {
            c = new T[nr_*nc_];
            mrank_ = std::min(nr_, nc_);
            size_ = nr_ * nc_;
            memcpy(c, m.c, nr_*nc_*sizeof(T));
        }
    }
    matrix(matrix &&m) : nr_(m.nr_), nc_(m.nc_), major_(m.major_)
    {
        c = m.c;
        mrank_ = m.mrank_;
        size_ = m.size_;
        m.nr_ = m.nc_ = 0;
        m.c = nullptr;
    }

    ~matrix()
    {
        nc_ = nr_ = mrank_ = size_ = 0;
        if (c)
        {
            delete [] c;
            c = nullptr;
        }
    }

    int nr() const { return nr_; }
    int nc() const { return nc_; }
    bool is_row_major() const { return major_ == MAJOR::ROW; }
    bool is_col_major() const { return major_ == MAJOR::COL; }
    const MAJOR& major() const { return major_; }
    int size() const { return size_; }
    void zero_out() { for (int i = 0; i < size(); i++) c[i] = 0.; }

    vec<T> get_row(int ir) const
    {
        if (ir < 0 || ir >= nr_) throw std::invalid_argument("out-of-bound row index");
        if (is_row_major())
            return vec<T>(nc_, c+ir*nc_);
        else
        {
            vec<T> v(nc_);
            for (int ic = 0; ic < nc_; ic++)
                v[ic] = this->at_by_cm(ir, ic);
        }
    }

    vec<T> get_col(int ic) const
    {
        if (ic < 0 || ic >= nc_) throw std::invalid_argument("out-of-bound col index");
        if (is_col_major())
            return vec<T>(nc_, c+ic*nr_);
        else
        {
            vec<T> v(nc_);
            for (int ir = 0; ir < nc_; ir++)
                v[ir] = this->at_by_cm(ir, ic);
        }
    }

    void random(const real_t &lb, const real_t &ub)
    {
        std::default_random_engine e(time(0));
        std::uniform_real_distribution<real_t> d(lb, ub);
        for (int i = 0; i < this->size(); i++)
            this->c[i] = d(e);
    }

    void set_diag(const T &v)
    {
        if (is_row_major())
        {
            for (int i = 0; i < mrank_; i++)
                this->at_by_rm(i, i) = v;
        }
        else
        {
            for (int i = 0; i < mrank_; i++)
                this->at_by_cm(i, i) = v;
        }
    }

    void swap_to_row_major()
    {
        if (is_col_major())
        {
            int nr = nr_, nc = nc_;
            transpose();
            reshape(nc, nr);
            major_ = MAJOR::ROW;
        }
    };
    void swap_to_col_major()
    {
        if (is_row_major())
        {
            int nr = nr_, nc = nc_;
            transpose();
            reshape(nr, nc);
            major_ = MAJOR::COL;
        }
    };

    // major independent matrix index
    T &operator()(const int ir, const int ic) { return this->at(ir, ic); }
    const T &operator()(const int ir, const int ic) const { return this->at(ir, ic); }
    T &at(const int ir, const int ic) { return c[flatid(nr_, ir, nc_, ic, is_row_major())]; }
    const T &at(const int ir, const int ic) const { return c[flatid(nr_, ir, nc_, ic, is_row_major())]; }
    // major dependent matrix index
    T &at_by_rm(const int ir, const int ic) { return c[flatid_rm(nr_, ir, nc_, ic)]; }
    const T &at_by_rm(const int ir, const int ic) const { return c[flatid_rm(nr_, ir, nc_, ic)]; }
    T &at_by_cm(const int ir, const int ic) { return c[flatid_cm(nr_, ir, nc_, ic)]; }
    const T &at_by_cm(const int ir, const int ic) const { return c[flatid_cm(nr_, ir, nc_, ic)]; }

    matrix<T> & operator=(const matrix<T> &m)
    {
        if (this == &m) return *this;
        resize(m.nr_, m.nc_);
        major_ = m.major_;
        memcpy(c, m.c, nr_*nc_*sizeof(T));
        return *this;
    }

    matrix<T> & operator=(matrix<T> &&m)
    {
        if (this == &m) return *this;
        nr_ = m.nr_;
        nc_ = m.nc_;
        mrank_ = m.mrank_;
        size_ = m.size_;
        major_ = m.major_;
        if(c) delete [] c;
        c = m.c;
        m.nr_ = m.nc_ = 0;
        m.c = nullptr;
        return *this;
    }

    matrix<T> & operator=(const std::vector<vec<T>> &nested_vec)
    {
        int nr_new, nc_new;
        get_nr_nc_from_nested_vec(nested_vec, nr_new, nc_new);
        resize(nr_new, nc_new);
        if (nr_&&nc_)
        {
            expand_nested_vector_to_pointer(nested_vec, nr_, nc_, c, is_row_major());
        }
        return *this;
    }

    matrix<T> & operator=(const T &cnum)
    {
        for (int i = 0; i < size(); i++)
            c[i] = cnum;
        return *this;
    }

    void operator+=(const matrix<T> &m)
    {
        assert(size() == m.size() && major() == m.major());
        for (int i = 0; i < size(); i++)
            c[i] += m.c[i];
    }

    void add_col(const std::vector<T> &v)
    {
        assert(nr_ == v.size());
        for (int ir = 0; ir < nr_; ir++)
            for (int ic = 0; ic < nc_; ic++)
                c[ir*nc_+ic] += v[ir];
    }

    void operator+=(const std::vector<T> &v)
    {
        assert(nc_ == v.size());
        for (int i = 0; i < nr_; i++)
            for (int ic = 0; ic < nc_; ic++)
                c[i*nc_+ic] += v[ic];
    }

    void operator+=(const vec<T> &v)
    {
        assert(nc_ == v.size());
        for (int i = 0; i < nr_; i++)
            for (int ic = 1; ic < nc_; ic++)
                c[i*nc_+ic] += v.c[ic];
    }

    void operator-=(const std::vector<T> &v)
    {
        assert(nc_ == v.size());
        for (int i = 0; i < nr_; i++)
            for (int ic = 1; ic < nc_; ic++)
                c[i*nc_+ic] -= v[ic];
    }

    void operator-=(const vec<T> &v)
    {
        assert(nc_ == v.size());
        for (int i = 0; i < nr_; i++)
            for (int ic = 1; ic < nc_; ic++)
                c[i*nc_+ic] -= v.c[ic];
    }

    void operator+=(const T &cnum)
    {
        for (int i = 0; i < size(); i++)
            c[i] += cnum;
    }

    void operator-=(const matrix<T> &m)
    {
        assert(size() == m.size() && major() == m.major());
        for (int i = 0; i < size(); i++)
            c[i] -= m.c[i];
    }
    void operator-=(const T &cnum)
    {
        for (int i = 0; i < size(); i++)
            c[i] -= cnum;
    }
    void operator*=(const T &cnum)
    {
        for (int i = 0; i < size(); i++)
            c[i] *= cnum;
    }
    void operator/=(const T &cnum)
    {
        assert(fabs(cnum) > 0);
        for (int i = 0; i < size(); i++)
            c[i] /= cnum;
    }

    void reshape(int nrows_new, int ncols_new)
    {
        assert ( size() == nrows_new * ncols_new);
        nr_ = nrows_new;
        nc_ = ncols_new;
        mrank_ = std::min(nr_, nc_);
    }

    void resize(int nrows_new, int ncols_new)
    {
        const int size_new = nrows_new * ncols_new;
        if (size_new)
        {
            if (c)
            {
                if ( size_new != size() )
                {
                    delete [] c;
                    c = new T[size_new];
                }
            }
            else
                c = new T[size_new];
        }
        else
        {
            if(c) delete [] c;
            c = nullptr;
        }
        nr_ = nrows_new;
        nc_ = ncols_new;
        mrank_ = std::min(nr_, nc_);
        size_ = nr_ * nc_;
        zero_out();
    }

    void resize(int nrows_new, int ncols_new, MAJOR major)
    {
        const int size_new = nrows_new * ncols_new;
        if (size_new)
        {
            if (c)
            {
                if ( size_new != size() )
                {
                    delete [] c;
                    c = new T[size_new];
                }
            }
            else
                c = new T[size_new];
        }
        else
        {
            if(c) delete [] c;
            c = nullptr;
        }
        nr_ = nrows_new;
        nc_ = ncols_new;
        mrank_ = std::min(nr_, nc_);
        major_ = major;
        size_ = nr_ * nc_;
        zero_out();
    }

    void conj() {};

    void transpose(bool conjugate = false, bool onsite_when_square_mat = true)
    {
        if (nr_ == nc_ && onsite_when_square_mat)
        {
            // handling within the memory of c pointer for square matrix
            int n = nr_;
            for (int i = 0; i != n; i++)
                for (int j = i+1; j < n; j++)
                {
                    T temp = c[i*n+j];
                    c[i*n+j] = c[j*n+i];
                    c[j*n+i] = temp;
                }
        }
        else
        {
            // NOTE: may have memory issue for large matrix as extra memory of c is required
            T* c_new = nullptr;
            if (c)
            {
                c_new = new T [nr_*nc_];
                if (is_row_major())
                {
                    for (int i = 0; i < nr_; i++)
                        for (int j = 0; j < nc_; j++)
                            c_new[flatid_rm(nc_, j, nr_, i)] = this->at_by_rm(i, j);
                }
                else
                {
                    for (int j = 0; j < nc_; j++)
                        for (int i = 0; i < nr_; i++)
                            c_new[flatid_cm(nc_, j, nr_, i)] = this->at_by_cm(i, j);
                }
                delete [] c;
            }
            c = c_new;
        }
        int temp = nc_;
        nr_ = temp;
        nc_ = nr_;
        if (conjugate) conj();
    }

    T det() const { return get_determinant(*this); }

    matrix<cplx_t> to_complex()
    {
        matrix<cplx_t> m(nr_, nc_, major_);
        for (int i = 0; i < size_; i++)
            m.c[i] = c[i];
        return m;
    }

    matrix<real_t> get_real()
    {
        matrix<real_t> m(nr_, nc_, major_);
        for (int i = 0; i < size_; i++)
            m.c[i] = get_real(c[i]);
        return m;
    }

    matrix<real_t> get_imag()
    {
        matrix<real_t> m(nr_, nc_, major_);
        for (int i = 0; i < size_; i++)
            m.c[i] = get_imag(c[i]);
        return m;
    }
};

template <> inline void matrix<std::complex<float>>::conj()
{
    for (int i = 0; i < size(); i++)
        c[i] = std::conj(c[i]);
}

template <> inline void matrix<std::complex<double>>::conj()
{
    for (int i = 0; i < size(); i++)
        c[i] = std::conj(c[i]);
}

template <typename T1, typename T2>
inline bool same_major(const matrix<T1> &m1, const matrix<T2> &m2)
{
    return m1.major() == m2.major();
}

template <typename T1, typename T2>
inline bool same_type(const matrix<T1> &m1, matrix<T2> &m2)
{
    return std::is_same<T1, T2>::value;
}

template <typename T1, typename T2>
void copy(const matrix<T1> &src, matrix<T2> &dest)
{
    assert(src.size() == dest.size() && same_major(src, dest));
    for (int i = 0; i < src.size(); i++)
        dest.c[i] = src.c[i];
}

template <typename T>
matrix<T> operator+(const matrix<T> &m1, const matrix<T> &m2)
{
    assert(m1.nc() == m2.nc() && m1.nr() == m2.nr() && same_major(m1, m2));
    matrix<T> sum = m1;
    sum += m2;
    return sum;
}

template <typename T>
inline matrix<T> operator+(const matrix<T> &m, const std::vector<T> &v)
{
    assert(m.nc() == v.size());
    matrix<T> mnew(m);
    mnew += v;
    return mnew;
}

template <typename T>
inline matrix<T> operator+(const matrix<T> &m, const T &cnum)
{
    matrix<T> sum = m;
    sum += cnum;
    return sum;
}

template <typename T>
inline matrix<T> operator+(const T &cnum, const matrix<T> &m)
{
    return m + cnum;
}

template <typename T>
inline matrix<T> operator-(const matrix<T> &m1, const matrix<T> &m2)
{
    assert(m1.nc() == m2.nc() && m1.nr() == m2.nr());
    matrix<T> mnew = m1;
    mnew -= m2;
    return mnew;
}

template <typename T>
inline matrix<T> operator-(const matrix<T> &m, const std::vector<T> &v)
{
    assert(m.nc() == v.size());
    matrix<T> mnew = m;
    mnew -= v;
    return mnew;
}

template <typename T>
inline matrix<T> operator-(const matrix<T> &m, const T &cnum)
{
    matrix<T> mnew = m;
    mnew -= cnum;
    return mnew;
}


template <typename T>
inline matrix<T> operator-(const T &cnum, const matrix<T> &m)
{
    return - m + cnum;
}

template <typename T>
inline matrix<T> operator*(const matrix<T> &m1, const matrix<T> &m2)
{
    assert(m1.nc() == m2.nr());
    assert(m1.major() == m2.major());

    auto major = m1.major();

    matrix<T> prod(m1.nr(), m2.nc(), major);
    if (m1.is_row_major())
    {
        linalg::gemm('N', 'N', m1.nr(), m2.nc(), m1.nc(),
                     1.0, m1.c, m1.nc(), m2.c, m2.nc(), 0.0, prod.c, prod.nc());
    }
    else
    {
        linalg::gemm_f('N', 'N', m1.nr(), m2.nc(), m1.nc(),
                       1.0, m1.c, m1.nr(), m2.c, m2.nr(), 0.0, prod.c, prod.nr());
    }

    return prod;
}

// a naive implementation for integer matrix, e.g. Spglib rotation matrices
template <>
inline matrix<int> operator*(const matrix<int> &m1, const matrix<int> &m2)
{
    assert(m1.nc() == m2.nr());
    matrix<int> prod(m1.nr(), m2.nc());
    for (int ir = 0; ir < m1.nr(); ir++)
        for (int ik = 0; ik < m1.nc(); ik++)
            for (int ic = 0; ic < m2.nc(); ic++)
                prod(ir, ic) += m1(ir, ik) * m2(ik, ic);
    return prod;
}

template <typename T>
inline vec<T> operator*(const matrix<T> &m, const vec<T> &v)
{
    assert(m.nc() == v.n);
    vec<T> mv(m.nr());
    if (m.is_row_major())
        linalg::gemv('N', m.nr(), m.nc(), 1.0, m.c, m.nc(), v.c, 1, 0.0, mv.c, 1);
    else
        linalg::gemv_f('N', m.nr(), m.nc(), 1.0, m.c, m.nr(), v.c, 1, 0.0, mv.c, 1);
    return mv;
}

template <typename T>
inline vec<T> operator*(const vec<T> &v, const matrix<T> &m)
{
    assert(m.nr() == v.n);
    vec<T> mv(m.nc());
    /* linalg::gemv('N', ); */
    if (m.is_row_major())
        linalg::gemv('T', m.nr(), m.nc(), 1.0, m.c, m.nc(), v.c, 1, 0.0, mv.c, 1);
    else
        linalg::gemv('T', m.nr(), m.nc(), 1.0, m.c, m.nr(), v.c, 1, 0.0, mv.c, 1);
    return mv;
}

template <typename T>
inline matrix<T> operator*(const matrix<T> &m, const T &cnum)
{
    matrix<T> sum = m;
    sum *= cnum;
    return sum;
}

template <typename T>
inline matrix<T> operator*(const T &cnum, const matrix<T> &m)
{
    return m * cnum;
}

template <typename T1, typename T2>
inline bool almost_equal(const matrix<T1> &m1, const matrix<T2> &m2, double thres = DOUBLE_EQUAL_THRES)
{
    int size = m1.size();
    if (size != m2.size() || size == 0) return false;
    if (m1.nc() != m2.nc() || m1.nr() != m2.nr() || !same_major(m1, m2)) return false;
    for (int i = 0; i < size; i++)
        if (fabs(m1.c[i] - m2.c[i]) > thres) return false;
    return true;
}

template <typename T1, typename T2>
inline bool almost_equal(const matrix<T1> &m, const T2 &cnum, double thres = DOUBLE_EQUAL_THRES)
{
    for (int i = 0; i < m.size(); i++)
        if (fabs(m.c[i] - cnum) > thres) return false;
    return true;
}

template <typename T1, typename T2>
inline bool operator==(const matrix<T1> &m1, const matrix<T2> &m2)
{
    return almost_equal(m1, m2, DOUBLE_EQUAL_THRES);
}

template <typename T1, typename T2>
inline bool operator!=(const matrix<T1> &m1, const matrix<T2> &m2)
{
    return !almost_equal(m1, m2, DOUBLE_EQUAL_THRES);
}

template <typename T1, typename T2>
inline bool operator==(const matrix<T1> &m, const T2 &cnum)
{
    return almost_equal(m, cnum, DOUBLE_EQUAL_THRES);
}

template <typename T1, typename T2>
inline bool operator!=(const matrix<T1> &m, const T2 &cnum)
{
    return !almost_equal(m, cnum, DOUBLE_EQUAL_THRES);
}

template <typename T>
inline matrix<T> inverse(const matrix<T> &m)
{
    if (m.size() == 0) throw std::invalid_argument("zero size matrix");
    matrix<T> inv(0, 0, m.major());
    inverse(m, inv);
    return inv;
}

template <typename T>
void inverse(const matrix<T> &m, matrix<T> &m_inv)
{
    assert(m.major() == m_inv.major());
    int lwork = m.nr();
    int info = 0;
    T work[lwork];
    int ipiv[std::min(m.nr(), m.nc())];
    m_inv.resize(m.nr(), m.nc());
    copy(m, m_inv);
    // debug
    // std::cout << m_inv.size() << " " << m_inv(0, 0) << " " << m_inv(m.nr-1, m.nc-1) << std::endl;
    if (m.is_row_major())
    {
        linalg::getrf(m_inv.nr(), m_inv.nc(), m_inv.c, m_inv.nc(), ipiv, info);
        linalg::getri(m_inv.nr(), m_inv.c, m_inv.nc(), ipiv, work, lwork, info);
    }
    else
    {
        linalg::getrf_f(m_inv.nr(), m_inv.nc(), m_inv.c, m_inv.nr(), ipiv, info);
        linalg::getri_f(m_inv.nr(), m_inv.c, m_inv.nr(), ipiv, work, lwork, info);
    }
    m_inv.reshape(m_inv.nc(), m_inv.nr());
}

template <typename T>
inline matrix<T> transpose(const matrix<T> &m, bool conjugate = false)
{
    matrix<T> mnew(m);
    mnew.transpose(conjugate);
    return mnew;
}

template <typename T>
std::ostream & operator<<(std::ostream & os, const matrix<T> &m)
{
    for (int i = 0; i < m.nr(); i++)
    {
        for (int j = 0; j < m.nc(); j++)
            os << m(i, j) << " ";
        os << std::endl;
    }
    return os;
}

template <typename T>
matrix<std::complex<T>> to_complex(const matrix<T> &m)
{
    matrix<std::complex<T>> cm(m.nr(), m.nc());
    for (int i = 0; i < m.size(); i++)
        cm.c[i] = m.c[i];
    return cm;
}

template <typename T>
matrix<T> get_real(const matrix<std::complex<T>> &cm)
{
    matrix<T> m(cm.nr(), cm.nc());
    for (int i = 0; i < cm.size(); i++)
        m.c[i] = cm.c[i].real();
    return m;
}

template <typename T>
matrix<T> get_imag(const matrix<std::complex<T>> &cm)
{
    matrix<T> m(cm.nr(), cm.nc());
    for (int i = 0; i < cm.size(); i++)
        m.c[i] = cm.c[i].imag();
    return m;
}

template <typename T>
T get_determinant(matrix<T> &m)
{
    int lwork = m.nr();
    int info = 0;
    int mrank = std::min(m.nr(), m.nc());
    T work[lwork];
    int ipiv[mrank];
    // debug
    if (m.is_row_major())
        linalg::getrf(m.nr(), m.nc(), m.c, m.nc(), ipiv, info);
    else
        linalg::getrf_f(m.nr(), m.nc(), m.c, m.nr(), ipiv, info);
    T det = 1;
    for (int i = 0; i < mrank; i++)
    {
        /* std::cout << i << " " << ipiv[i] << " " << m.c[i*m.nc+i] << " "; */
        det *= (2*int(ipiv[i] == (i+1))-1) * m(i, i);
    }
    return det;
}

template <typename T>
matrix<double> to_double(const matrix<T> &mat)
{
    matrix<double> dmat(mat.nr(), mat.nc());
    if (dmat.size())
    {
        for (int ir = 0; ir < mat.nr(); ir++)
            for (int ic = 0; ic < mat.nc(); ic++)
                dmat(ir, ic) = mat(ir, ic);
    }
    return dmat;
}

template <typename T>
T maxabs(const matrix<std::complex<T>> &cmat)
{
    T ma = 0.;
    for (int i = 0; i < cmat.size(); i++)
    {
        T tmp = fabs(cmat.c[i]);
        if (tmp > ma) ma = tmp;
    }
    return ma;
}

template <typename T>
T maxabs(const matrix<T> &cmat)
{
    T ma = 0.;
    for (int i = 0; i < cmat.size(); i++)
    {
        T tmp = fabs(cmat.c[i]);
        if (tmp > ma) ma = tmp;
    }
    return ma;
}

template <typename T>
std::string str(const matrix<T> &m)
{
    std::string s;
    for (int i = 0; i < m.nr(); i++)
    {
        if (m.nc() > 0) s = s + std::to_string(m(i, 0));
        for (int j = 1; j < m.nc(); j++)
            s = s + " " + std::to_string(m(i, j));
        s = s + "\n";
    }
    return s;
}

template <typename T>
std::string str(const matrix<std::complex<T>> &m)
{
    std::string s;
    for (int i = 0; i != m.nr(); i++)
    {
        if (m.nc() > 0)
            s = s + "(" + std::to_string(m(i, 0).real()) + "," + std::to_string(m(i, 0).imag()) + ")";
        for (int j = 1; j != m.nc(); j++)
            s = s + " (" + std::to_string(m(i, j).real()) + "," + std::to_string(m(i, j).imag()) + ")";
        s = s + "\n";
    }
    return s;
}

template <typename T>
inline matrix<T> random(int nr, int nc, const T &lb, const T &ub)
{
    matrix<T> m(nr, nc);
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<T> d(lb, ub);
    for (int i = 0; i != m.size(); i++)
        m.c[i] = d(e);
}

template <typename T>
inline matrix<std::complex<T>> random(int nr, int nc, const std::complex<T> &lb, const std::complex<T> &ub)
{
    matrix<std::complex<T>> m(nr, nc);
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<T> dr(lb.real(), ub.real()), di(lb.imag(), lb.imag());
    for (int i = 0; i != m.size(); i++)
        m.c[i] = std::complex<T>{dr(e), di(e)};
}

template <typename T>
inline matrix<T> random_sy(int n, const T &lb, const T &ub)
{
    matrix<T> m(n, n);
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<T> d(lb, ub);
    for (int i = 0; i != n; i++)
    {
        for (int j = i; j != n; j++)
        {
            m(i, j) = m(j, i) = d(e);
        }
    }
    return m;
}

template <typename T>
inline matrix<std::complex<T>> random_he(int n, const std::complex<T> &lb, const std::complex<T> &ub)
{
    matrix<std::complex<T>> m(n, n);
    std::default_random_engine e(time(0));
    std::uniform_real_distribution<T> dr(lb.real(), ub.real()), di(lb.imag(), lb.imag());
    for (int i = 0; i != n; i++)
    {
        m(i, i) = dr(e);
        for (int j = i + 1; j != n; j++)
        {
            m(i, j) = std::conj(m(j, i) = std::complex<T>{dr(e), di(e)});
        }
    }
    return m;
}

/* ========================== */
/* ScaLAPACK related utitlies */
/* ========================== */

template <typename T>
inline matrix<T> init_local_mat(const ArrayDesc &ad, MAJOR major)
{
    matrix<T> mat_lo(ad.num_r(), ad.num_c(), major);
    return mat_lo;
}

template <typename T>
inline matrix<T> get_local_mat(const matrix<T> &mat_go, const ArrayDesc &ad, MAJOR major)
{
    // assert the shape of matrix conforms with the array descriptor
    assert(mat_go.nr() == ad.m() && mat_go.nc() == ad.n());

    matrix<T> mat_lo(ad.num_r(), ad.num_c(), major);
    for (int i = 0; i != mat_go.nr(); i++)
    {
        auto i_lo = ad.to_loid_r(i);
        if (i_lo < 0) continue;
        for (int j = 0; j != mat_go.nc(); j++)
        {
            auto j_lo = ad.to_loid_c(j);
            if (j_lo < 0) continue;
            mat_lo(i_lo, j_lo) = mat_go(i, j);
        }
    }
    return mat_lo;
}

template <typename T>
inline matrix<T> get_local_mat_pcoord(const matrix<T> &mat_go,
                                      const int &mb, const int &nb,
                                      const int &irsrc, const int &icsrc,
                                      const int &nprows, const int &myprow,
                                      const int &npcols, const int &mypcol,
                                      MAJOR major)
{
    int m = mat_go.nr(), n = mat_go.nc();
    int m_local = linalg::numroc(m, mb, myprow, irsrc, nprows);
    int n_local = linalg::numroc(n, nb, mypcol, icsrc, npcols);
    matrix<T> mat_lo(m_local, n_local, major);
    for (int i = 0; i != m; i++)
    {
        auto i_lo = local_index(i, m, mb, nprows, myprow);
        if (i_lo < 0) continue;
        for (int j = 0; j != n; j++)
        {
            auto j_lo = local_index(j, n, nb, npcols, mypcol);
            if (j_lo < 0) continue;
            mat_lo(i_lo, j_lo) = mat_go(i, j);
        }
    }
    return mat_lo;
}

//! wrapper of the above template, using process grid and layout of BLACS_handler
template <typename T>
inline matrix<T> get_local_mat_pid(const matrix<T> &mat_go,
                                   const int &mb, const int &nb,
                                   const int &irsrc, const int &icsrc,
                                   const BLACS_handler &blacs_h, const int &pid,
                                   MAJOR major)
{
    auto pcoord = get_pcoord_from_pid(pid, blacs_h.nprows, blacs_h.npcols, blacs_h.layout);
    return get_local_mat_pcoord(mat_go, mb, nb, irsrc, icsrc,
                                blacs_h.nprows, pcoord.first, blacs_h.npcols, pcoord.second,
                                major);
}

template <typename T1, typename T2>
void get_local_mat(matrix<T1> &mat_lo, const matrix<T2> &mat_go, const ArrayDesc &ad)
{
    // assert the shape of matrix conforms with the array descriptor
    assert(mat_go.nr() == ad.m() && mat_go.nc() == ad.n());
    assert(mat_lo.nr() == ad.num_r() && mat_lo.nc() == ad.num_c());

    for (int i = 0; i != mat_go.nr(); i++)
    {
        auto i_lo = ad.to_loid_r(i);
        if (i_lo < 0) continue;
        for (int j = 0; j != mat_go.nc(); j++)
        {
            auto j_lo = ad.to_loid_c(j);
            if (j_lo < 0) continue;
            mat_lo(i_lo, j_lo) = mat_go(i, j);
        }
    }
}

