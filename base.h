#pragma once
#include <complex>
#include <cmath>

constexpr static const double DOUBLE_EQUAL_THRES = 1e-10;

template <typename T, typename T1, typename T2>
T norm(const T v[], const T1 &n, const T2 power)
{
    T norm = 0;
    for (int i = 0; i < n; i++)
        norm += std::pow(std::fabs(v[i]), power);
    return std::pow(norm, 1./power);
}

template <typename Tv, typename T1, typename T2>
Tv norm(const std::complex<Tv> v[], const T1 &n, const T2 power)
{
    Tv norm = 0;
    for (int i = 0; i < n; i++)
        norm += std::pow(std::fabs(v[i]), power);
    return std::pow(norm, 1./power);
}

// check if passed template type argument is a complex value
// from https://stackoverflow.com/questions/41438493/how-to-identifying-whether-a-template-argument-is-stdcomplex
template <typename T>
struct is_complex_t: public std::false_type {};

template <typename T>
struct is_complex_t<std::complex<T>>: public std::true_type {};

template <typename T>
constexpr bool is_complex() { return is_complex_t<T>::value; }

template <typename T>
struct to_real { using type = T; };
template <typename T>
struct to_real<std::complex<T>> { using type = T; };

template <typename T>
struct to_cplx { using type = std::complex<T>; };
template <typename T>
struct to_cplx<std::complex<T>> { using type = T; };
