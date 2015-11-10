/* File:   dumpASCII.h */
/* Date:   Tue Nov 10 10:08:43 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Simple dump to text for e.g. Gnuplot */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef DUMPASCII_H_COXPBI4D
#define DUMPASCII_H_COXPBI4D

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

#endif /* DUMPASCII_H_COXPBI4D */
