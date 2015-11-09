/*****************************************************************************
 * 
 * (Discrete) 3D FDTD solver
 *
 *****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


// chose the type for the field: int, float, double, .. 
#define FLDTYPE double

const int ncomp = 3;            // number of components in each field
const int compX = 0;            // index of the X component 
const int compY = 1;            // index of the Y component
const int compZ = 2;            // index of the Z component


//const int nz = 256; 

long currentTimeMillis();
long currentTimeMicros();
void write_file(char *filename, FLDTYPE* fld, int nx, int ny, int nz);

	 
FLDTYPE* e;                      // e field
FLDTYPE* b;                      // b field


int main(int argc, char* argv[]){

  long startTime, midTime0, midTime1, midTime2, endTime;

  int nx = 128;             // domain size
  int ny = 128; 
  int nz = 128; 
  int nsteps = 100;         // number of time steps
  
  FLDTYPE LX = .000016;             // domain size
  FLDTYPE LY = .00005; 
  FLDTYPE LZ = .00005; 

  FLDTYPE XSTART = 0.;             // start position
  FLDTYPE YSTART = -LY/2.0; 
  FLDTYPE ZSTART = -LZ/2.0;

  if (argc>0) {
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    nz = atoi(argv[3]);
    nsteps = atoi(argv[4]);
  }
  FLDTYPE dx = LX/nx;             // grid spacing in x
  FLDTYPE dy = LY/ny;             // grid spacing in y
  FLDTYPE dz = LZ/nz;             // grid spacing in z
  int nxleftover=(int) floor(nx/4);
  e = (FLDTYPE *) malloc(ncomp * nx * ny * nz * sizeof(FLDTYPE));
  b = (FLDTYPE *) malloc(ncomp * nx * ny * nz * sizeof(FLDTYPE));
  long tot = 0;

  //printf("Begin iteration \n");
  //printf("nxleftover=%d\n",nxleftover);

  // some constants for the radiating boundary conditions
  const FLDTYPE ky = 5 * 3.14159 / (FLDTYPE) ny;
  const FLDTYPE kz = 8 * 3.14159 / (FLDTYPE) nz;
  FLDTYPE dt;
  const FLDTYPE omega = 5e4;
  const FLDTYPE odt = 0.02 * 3.14159;
  const FLDTYPE c = 2.9979e8;
  const int cdx4 = 2;
  FLDTYPE dtOverDx, dtOverDy, dtOverDz;
  FLDTYPE cdtOverDx, cdtOverDy, cdtOverDz;
  FLDTYPE dxi=1/dx;
  FLDTYPE dyi=1/dy;
  FLDTYPE dzi=1/dz;
  FLDTYPE DLI = sqrt(dxi*dxi + dyi*dyi + dzi*dzi);
  FLDTYPE DL = 1./DLI;
  dt =  0.995 * DL / c;
  
  dtOverDx = dt/dx;
  dtOverDy = dt/dy;
  dtOverDz = dt/dz;

  cdtOverDx = c*dt/dx;
  cdtOverDy = c*dt/dy;
  cdtOverDz = c*dt/dz;

  // *******************
  // begin of the time stepping loop 
  // *******************

  for(int n = 0; n < ncomp * nx * ny * nz; n++){
    e[n]=0.0;
    b[n]=0.0;
  }
  // just some timing ...
  startTime = currentTimeMillis();


  for(int n = 0; n < nsteps; n++){

    // Set the boundary condition at each timestep.
    // The following generates a laser pulse entering from the left. In 
    // a more realistic code, this might be a more complicated function.. 
    for(int z = 0; z < nz; z++)
      for(int y =0; y < ny; y++) {
	int ind = (y * nx + z * nx * ny) * ncomp;
	e[ind+compZ] = (FLDTYPE) 1.e8;
      }
    

    // B field update. Assuming fully periodic boundary conditions.
#ifdef LOOPTRICK
    for(int z = 0; z < nz; z++)
      for(int y =0; y < ny; y++) {
	for(int x=0; x < nxleftover*4; x+=4)
	  for (int xx=0; xx<4; xx++) {
	    int ind = (x + xx + y * nx + z * nx * ny) * ncomp;
	    int ind_px = ((x + xx + 1) % nx + y * nx + z * nx * ny) * ncomp;
	    int ind_py = (x + xx+((y + 1) % ny) * nx + z * nx * ny) * ncomp;
	    int ind_pz = (x + xx + y*nx + (( z + 1) % nz) * nx * ny) * ncomp;
#ifndef INDEXONLY
	    b[ind + compX] -=  (e[ind_py + compZ] - e[ind + compZ])*dtOverDy
	      - (e[ind_pz + compY] - e[ind + compY])*dtOverDz;
	    
	    b[ind + compY] -=  (e[ind_pz + compX] - e[ind + compX])*dtOverDz
	      - (e[ind_px + compZ] - e[ind + compZ])*dtOverDx;
	    
	    b[ind + compZ] -=  (e[ind_px + compY] - e[ind + compY])*dtOverDx
	      - (e[ind_py + compX] - e[ind + compX])*dtOverDy;	    
#else
	    tot+=ind+ind_px+ind_py+ind_pz;
#endif
	  }


    	for(int x=(nxleftover)*4; x < nx; x++) {
	  int ind = (x + y * nx + z * nx * ny) * ncomp;
	  int ind_px = ((x + 1) % nx + y * nx + z * nx * ny) * ncomp;
	  int ind_py = (x+((y + 1) % ny) * nx + z * nx * ny) * ncomp;
	  int ind_pz = (x+ y*nx + (( z + 1) % nz) * nx * ny) * ncomp;	  
#ifndef INDEXONLY
	  b[ind + compX] -=  (e[ind_py + compZ] - e[ind + compZ])*dtOverDy
	    - (e[ind_pz + compY] - e[ind + compY])*dtOverDz;
	  
	  b[ind + compY] -=  (e[ind_pz + compX] - e[ind + compX])*dtOverDz
	    - (e[ind_px + compZ] - e[ind + compZ])*dtOverDx;
	  
	  b[ind + compZ] -=  (e[ind_px + compY] - e[ind + compY])*dtOverDx
	      - (e[ind_py + compX] - e[ind + compX])*dtOverDy;	    
#else
	  tot+=ind+ind_px+ind_py+ind_pz;
#endif
	}
      }
    
    
    // E field update. Assuming fully periodic boundary conditions.
    for(int z = 0; z < nz; z++)
      for(int y =0; y < ny; y++) {
	for(int x=0; x < nxleftover*4; x+=4)
	  for (int xx=0; xx<4; xx++) {
	    int ind = (x + xx + y * nx + z * nx * ny) * ncomp;
	    int ind_mx = ((x + xx - 1 + nx) % nx + y * nx + z * nx * ny) * ncomp;
	    int ind_my = (x + xx+((y - 1 + ny) % ny) * nx + z * nx * ny) * ncomp;
	    int ind_mz = (x + xx+ y*nx + (( z - 1 + nz) % nz) * nx * ny) * ncomp;	  
#ifndef INDEXONLY
	    e[ind + compX] += (b[ind + compZ] - b[ind_my + compZ]) * cdtOverDy 
	      - (b[ind + compY] - b[ind_mz + compY]) * cdtOverDz;
	    e[ind + compY] += (b[ind + compX] - b[ind_mz + compX]) * cdtOverDz
	      - (b[ind + compZ] - b[ind_mx + compZ]) * cdtOverDx;
	    e[ind + compZ] += (b[ind + compY] - b[ind_mx + compY]) * cdtOverDx 
	      - (b[ind + compX] - b[ind_my + compX]) * cdtOverDy;
#else
	    tot+=ind+ind_mx+ind_my+ind_mz;
#endif
	  }

    	for(int x=(nxleftover)*4; x < nx; x++) {
	  int ind = (x + y * nx + z * nx * ny) * ncomp;
	  int ind_mx = ((x - 1 + nx) % nx + y * nx + z * nx * ny) * ncomp;
	  int ind_my = (x+((y - 1 + ny) % ny) * nx + z * nx * ny) * ncomp;
	  int ind_mz = (x+ y*nx + (( z - 1 + nz) % nz) * nx * ny) * ncomp;	  
#ifndef INDEXONLY
	  e[ind + compX] += (b[ind + compZ] - b[ind_my + compZ]) * cdtOverDy 
	    - (b[ind + compY] - b[ind_mz + compY]) * cdtOverDz;
	  e[ind + compY] += (b[ind + compX] - b[ind_mz + compX]) * cdtOverDz
	    - (b[ind + compZ] - b[ind_mx + compZ]) * cdtOverDx;
	  e[ind + compZ] += (b[ind + compY] - b[ind_mx + compY]) * cdtOverDx 
	    - (b[ind + compX] - b[ind_my + compX]) * cdtOverDy;
#else
	  tot+=ind+ind_mx+ind_my+ind_mz;
#endif
	}
      }

    
#else ///Not Using LOOPTRICK
    for(int z = 0; z < nz; z++) {
      for(int y =0; y < ny; y++)
	for(int x=0; x < nx; x++) {
	  int ind = (x+ y * nx + z * nx * ny) * ncomp;
	  int ind_px = ((x+ 1) % nx + y * nx + z * nx * ny) * ncomp;
	  int ind_py = (x+((y + 1) % ny) * nx + z * nx * ny) * ncomp;
	  int ind_pz = (x+ y*nx + (( z + 1) % nz) * nx * ny) * ncomp;	      
#ifndef INDEXONLY
	  b[ind + compX] -=  (e[ind_py + compZ] - e[ind + compZ])*dtOverDy
	    - (e[ind_pz + compY] - e[ind + compY])*dtOverDz;
	  
	  b[ind + compY] -=  (e[ind_pz + compX] - e[ind + compX])*dtOverDz
	    - (e[ind_px + compZ] - e[ind + compZ])*dtOverDx;
	  
	  b[ind + compZ] -=  (e[ind_px + compY] - e[ind + compY])*dtOverDx
	    - (e[ind_py + compX] - e[ind + compX])*dtOverDy;  
#else	
	  tot+=ind+ind_px+ind_py+ind_pz;
#endif
	}
    }
    // E field update. Assuming fully periodic boundary conditions.
    for(int z = 0; z < nz; z++)
      for(int y =0; y < ny; y++)
	for(int x=0; x < nx; x++) {	  
	  int ind = (x + y * nx + z * nx * ny) * ncomp;
          int ind_mx = ((x - 1 + nx) % nx + y * nx + z * nx * ny) * ncomp;
          int ind_my = (x+((y - 1 + ny) % ny) * nx + z * nx * ny) * ncomp;
          int ind_mz = (x+ y*nx + (( z - 1 + nz) % nz) * nx * ny) * ncomp;
#ifndef INDEXONLY
          e[ind + compX] += (b[ind + compZ] - b[ind_my + compZ]) * cdtOverDy 
	    - (b[ind + compY] - b[ind_mz + compY]) * cdtOverDz;
	  e[ind + compY] += (b[ind + compX] - b[ind_mz + compX]) * cdtOverDz
	    - (b[ind + compZ] - b[ind_mx + compZ]) * cdtOverDx;
          e[ind + compZ] += (b[ind + compY] - b[ind_mx + compY]) * cdtOverDx 
	    - (b[ind + compX] - b[ind_my + compX]) * cdtOverDy;
#else
	  tot+=ind+ind_mx+ind_my+ind_mz;
#endif
	}

#endif

  }


  endTime = currentTimeMillis();
#ifdef INDEXONLY
  printf("tot= %d\n", tot );
#endif
// at this point, we can dump the fields for later processing

#ifdef LOOPTRICK
  printf("discrete-LOOPTRICK : grid=(%d,%d,%d), timeSteps=%d : Runtime = %g\n", nx,ny,nz,nsteps,(double)(endTime - startTime) / 1000. );
  write_file("efield-LOOPTRICK.txt",e, nx, ny, nz);
  write_file("bfield-LOOPTRICK.txt",b, nx, ny, nz);
#else
  printf("discrete : grid=(%d,%d,%d), timeSteps=%d : Runtime = %g\n", nx,ny,nz,nsteps,(double)(endTime - startTime) / 1000. );
  write_file("efield.txt", e, nx, ny, nz);
  write_file("bfield.txt", b, nx, ny, nz);
#endif

  free(e);
  free(b);
}



//
// write a field to the file. Some primitive serialization is done in 
// case of a parallel run.
//
void write_file(char *filename, FLDTYPE* fld, int nx, int ny, int nz){
  FILE* outfile;

  outfile = fopen(filename, "w");
  for(int k = 0; k < nx * ny * nz * ncomp; k+=3){
    fprintf(outfile, "(%d,%d,%d)\t(%g,%g,%g)\n", k/ncomp % (nx),k/ncomp/nx % (ny),k/(nx*ny)/ncomp % nz, fld[k], fld[k+1], fld[k+2]);
    //fprintf(outfile, "(%d,%d,%d)\t(%g,%g,%g)\n", k/ncomp % (nx),k/ncomp/nx % (ny),k/(nx*ny)/ncomp % nz,fabs(fld[k]) < 1.e-7 ? 0 : fld[k],  fabs(fld[k+1]) < 1.e-7 ? 0 : fld[k+1],  fabs(fld[k+2]) < 1.e-7 ? 0 : fld[k+2]);
    //for(int k = 0; k < nx * ny * nz * ncomp; k++){
    //fprintf(outfile, "%g\n", fld[k]);
   // fprintf(outfile, "%g\n", fld[k]);
  }
  fclose(outfile);
}

//
// just some timer for performance menasurement
//
#include <sys/time.h>
#include <unistd.h>

long currentTimeMillis() {

    struct timeval tv ;
    struct timezone tz ;

    gettimeofday(&tv, &tz) ;

    return tv.tv_sec * 1000 + tv.tv_usec / 1000 ;
}

long currentTimeMicros() {

    struct timeval tv ;
    struct timezone tz ;

    gettimeofday(&tv, &tz) ;

    return tv.tv_sec * 1000000 + tv.tv_usec / 1000000 ;
}
