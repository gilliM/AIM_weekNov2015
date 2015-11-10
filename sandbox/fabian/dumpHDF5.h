/* File:   dumpHDF5.h */
/* Date:   Tue Nov 10 07:34:12 2015 */
/* Author: Fabian Wermelinger */
/* Tag:    Dump HDF5 data */
/* Copyright 2015 ETH Zurich. All Rights Reserved. */
#ifndef DUMPHDF5_H_YI0S95GY
#define DUMPHDF5_H_YI0S95GY

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>
#ifdef _USE_HDF_
#include <hdf5.h>
#endif

#ifdef _FLOAT_PRECISION_
typedef float Real;
#else
typedef double Real;
#endif

#ifdef _FLOAT_PRECISION_
#define HDF_REAL H5T_NATIVE_FLOAT
#else
#define HDF_REAL H5T_NATIVE_DOUBLE
#endif


template<size_t _NX, size_t _NY, size_t _NZ>
void dumpHDF5(const std::string& fname, const std::vector<Real>& u, const Real t)
{
#ifdef _USE_HDF_
    char filename[256];
    herr_t status;
    hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

    hsize_t count[4]  = {_NZ, _NY, _NX, 1};
    hsize_t dims[4]   = {_NZ, _NY, _NX, 1};
    hsize_t offset[4] = {0, 0, 0, 0};
    sprintf(filename, "%s.h5", fname.c_str());

    H5open();
    fapl_id    = H5Pcreate(H5P_FILE_ACCESS);
    file_id    = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    status     = H5Pclose(fapl_id);
    fapl_id    = H5Pcreate(H5P_DATASET_XFER);
    fspace_id  = H5Screate_simple(4, dims, NULL);
    dataset_id = H5Dcreate(file_id, "data", HDF_REAL, fspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    fspace_id  = H5Dget_space(dataset_id);

    H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
    mspace_id = H5Screate_simple(4, count, NULL);

    status = H5Dwrite(dataset_id, HDF_REAL, mspace_id, fspace_id, fapl_id, u.data());

    status = H5Sclose(mspace_id);
    status = H5Sclose(fspace_id);
    status = H5Dclose(dataset_id);
    status = H5Pclose(fapl_id);
    status = H5Fclose(file_id);
    H5close();
    {
        char wrapper[256];
        sprintf(wrapper, "%s.xmf", fname.c_str());
        FILE *xmf = 0;
        xmf = fopen(wrapper, "w");
        fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
        fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
        fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
        fprintf(xmf, " <Domain>\n");
        fprintf(xmf, "   <Grid GridType=\"Uniform\">\n");
        fprintf(xmf, "     <Time Value=\"%05e\"/>\n", t);
        fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n", (int)dims[0], (int)dims[1], (int)dims[2]);
        fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
        fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        fprintf(xmf, "        %e %e %e\n", 0.,0.,0.);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
        fprintf(xmf, "        %e %e %e\n", 1./(Real)dims[0],1./(Real)dims[0],1./(Real)dims[0]);
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Geometry>\n");

        fprintf(xmf, "     <Attribute Name=\"data\" AttributeType=\"%s\" Center=\"Node\">\n", "Scalar");
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n", (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3]);
        fprintf(xmf, "        %s:/data\n",(fname+".h5").c_str());
        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");

        fprintf(xmf, "   </Grid>\n");
        fprintf(xmf, " </Domain>\n");
        fprintf(xmf, "</Xdmf>\n");
        fclose(xmf);
    }
#endif
}

#endif /* DUMPHDF5_H_YI0S95GY */
