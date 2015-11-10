/* File:   acoustic2D.cpp */
/* Date:   Mon Nov  9 22:18:01 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Serial 2D acoustic solver */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#include <cassert>
#include <vector>
#include <cstddef>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <omp.h>

#include "ArgumentParser.h"
#include "dumpHDF5.h"

using namespace std;

#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif

#ifndef _NX_
#define _NX_ 256
#endif /* _NX_ */
#ifndef _NY_
#define _NY_ 256
#endif /* _NY_ */


///////////////////////////////////////////////////////////////////////////////
// helpers
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


template <size_t _NX, size_t _NY>
void _dumpASCII(const string& fname, const vector<Real>& u, const Real t)
{
    ofstream o(fname.c_str(), std::ios::out);
    o << "# Time = " << t << endl;
    for (size_t j = 0; j < _NY; ++j)
    {
        for (size_t i = 0; i < _NX; ++i)
            o << u[i + j*_NX] << endl;
        o << endl;
    }
    o.close();
}
///////////////////////////////////////////////////////////////////////////////


int main(int argc, char** argv)
{
    ArgumentParser parser(argc, (const char**)argv);

    const bool bIO = parser("-IO").asBool(true);

    // global constants
    const size_t NX = _NX_; // x-nodes
    const size_t NY = _NY_; // y-nodes

    // wave constants
    const Real gc = parser("-gc").asDouble(100.0);
    const Real gk = parser("-gk").asDouble(6.28/1000.);
    const Real gw = gk*gc;

    // simulation parameter
    const Real dx2 = parser("-dx2").asDouble(1.0);
    const Real dy2 = parser("-dy2").asDouble(1.0);
    const Real dt  = parser("-dt").asDouble(0.01);
    const Real tEnd = parser("-tEnd").asDouble(100.0);

    vector<Real> vel(NX*NY); // velocity
    vector<Real> amp(NX*NY); // amplitude

    // NUMA first touch
#pragma omp parallel for
    for (size_t j = 0; j < NY; ++j)
        for (size_t i = 0; i < NX; ++i)
        {
            vel[i + j*NX] = 0.0;
            amp[i + j*NX] = 0.0;
        }
    /* amp[2+2*NX] = 1.0; */

    Real t = 0.0;
    size_t step = 0;

    // main loop over time
    while (t < tEnd)
    {
        const Real dtMax = (tEnd-t) < dt ? (tEnd-t) : dt;

        // solve acoustic PDE
        _updateAmplitude<NX,NY>(amp, vel, dtMax);
        _updateVelocity<NX,NY>(vel, amp, dx2, dy2, dtMax, gc);
        t += dtMax;
        ++step;

        // dump stuff
        if (bIO && (step%10 == 0))
        {
            ostringstream fname;
            fname << "amplitude_" << setw(5) << setfill('0') << step << ".dat";
            if (parser.check("-hdf"))  // dump HDF5
                dumpHDF5<NX,NY,1>(fname.str(), amp, t);
            else // ASCII else
                _dumpASCII<NX,NY>(fname.str(), amp, t);
        }

        // update boundary
        _boundaryCondition<NX,NY>(amp, t, gk, gw);
    }

    return 0;
}
