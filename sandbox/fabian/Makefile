CC = g++

.PHONY = clean

all: acoustic2D

acoustic2D: acoustic2D.cpp
	$(CC) -fopenmp -g -O3 -DNDEBUG -D_USE_HDF_ acoustic2D.cpp -lhdf5 -o acoustic2D

clean:
	rm -rf acoustic2D *~ *.h5 *.xmf *.dat *.dSYM
