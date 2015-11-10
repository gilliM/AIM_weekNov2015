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
#include "dumpASCII.h"

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
// Compute Kernels to use (OpenMP, CUDA, OpenACC, etc)
///////////////////////////////////////////////////////////////////////////////
#include "openMP_kernels.h"
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
            fname << "amplitude_" << setw(6) << setfill('0') << step << ".dat";
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
