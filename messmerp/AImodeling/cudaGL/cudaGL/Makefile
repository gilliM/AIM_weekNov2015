all: cudaGL


cudaGL: cudaGL.cu
	nvcc  -arch=sm_20 -I/project/g33/messmerp/glew/glew/include/ -I/project/g33/messmerp/freeglut/freeglut/include -L/project/g33/messmerp/glew/glew/lib64 -L/project/g33/messmerp/freeglut/freeglut/lib -L/opt/cray/nvidia/default/lib64 -o cudaGL cudaGL.cu  -lglut  -lGLEW -lglut -lGLU -lGL

clean:
	rm cudaGL cudaGL.obj


