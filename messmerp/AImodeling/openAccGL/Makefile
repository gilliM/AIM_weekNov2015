
# openACC version of the wave propagation sample.
# currently requires PGI OpenACC compiler

all: openAccGL


openAccGL: openAccGL.c
	cc -acc -I/project/g33/messmerp/glew/glew/include/ -I/project/g33/messmerp/freeglut/freeglut/include -L/project/g33/messmerp/glew/glew/lib64 -L/project/g33/messmerp/freeglut/freeglut/lib -L/opt/cray/nvidia/default/lib64 -o openAccGL openAccGL.c  -lglut  -lGLEW -lglut -lGLU -lGL
#	cc -I/project/g33/messmerp/glew/glew/include/ -I/project/g33/messmerp/freeglut/freeglut/include -L/project/g33/messmerp/glew/glew/lib64 -L/project/g33/messmerp/freeglut/freeglut/lib -L/opt/cray/nvidia/default/lib64 -o openAccGL openAccGL.c  -lglut  -lGLEW -lglut -lGLU -lGL

clean:
	rm openAccGL


