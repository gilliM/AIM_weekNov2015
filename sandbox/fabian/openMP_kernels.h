/* File:   openMP_kernels.h */
/* Date:   Tue Nov 10 10:06:42 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    OpenMP kenrnels */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef OPENMP_KERNELS_H_FASZ9OMS
#define OPENMP_KERNELS_H_FASZ9OMS

template <size_t _NX, size_t _NY>
void _updateAmplitude(vector<Real>& u, const vector<Real>& v, const Real dt)
{
    assert(u.size() == _NX*_NY);
    assert(u.size() == v.size());

    // point-wise update
#pragma omp parallel for
    for (size_t j = 0; j < _NY; ++j)
        for (size_t i = 0; i < _NX; ++i)
            u[i + j*_NX] += dt*v[i + j*_NX];
}

template <size_t _NX, size_t _NY>
void _updateVelocity(vector<Real>& v, const vector<Real>& u, const Real dx2, const Real dy2, const Real dt, const Real gc)
{
    assert(u.size() == _NX*_NY);
    assert(u.size() == v.size());

    // stencil update
#pragma omp parallel for
    for (size_t j = 1; j < _NY-1; ++j)
        for (size_t i = 1; i < _NX-1; ++i)
        {
            const size_t i0  = i + j*_NX;
            const size_t ip1 = (i+1) + j*_NX;
            const size_t im1 = (i-1) + j*_NX;
            const size_t jp1 = i + (j+1)*_NX;
            const size_t jm1 = i + (j-1)*_NX;

            v[i + j*_NX] += dt*gc*(
                    (u[jp1] - static_cast<Real>(2.0)*u[i0] + u[jm1])/dx2 +
                    (u[ip1] - static_cast<Real>(2.0)*u[i0] + u[im1])/dy2);
        }
}


template <size_t _NX, size_t _NY>
void _boundaryCondition(vector<Real>& u, const Real t, const Real gk, const Real gw)
{
    /* for (size_t j = 0; j < 1; ++j) */
    for (size_t i = 0; i < _NX/4; ++i)
        u[i] = sin(gw*t + gk*i)*cos(gw*t + gk*0);

    for (size_t j = 0; j < _NY/4; ++j)
        /* for (size_t i = 0; i < 1; ++i) */
        u[0 + j*_NX] = sin(gw*t + gk*j)*cos(gw*t + gk*0);
}

#endif /* OPENMP_KERNELS_H_FASZ9OMS */
