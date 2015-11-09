/*
 * Sample for OpenACC - OpenGL interoperability
 *
 * This sample runs a basic finite-difference solver of the 2D scalar wave equation. 
 *
 * usage: at program start, the sample launches a wave from the lower left corner into 
 *        the domain. Various hot keys are recognized:
 *        ' ' : restart the simulation
 *        'l' : add interior boundary condition ('logo')
 *        'n' : remove interior boundary condition ('no logo')
 *       'ESC': quit
 *       In addition, the user can interact with the scene via mouse (rotate and scale)
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> 

#include <inttypes.h>



// OpenGL Graphics includes
#include <GL/glew.h>
#include <GL/freeglut.h>

#define __align__(x)
#define CUDARTAPI
#define __location__(a)

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"

#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants

// we want to be able to compile with a plain C compiler, so we have to 
// resort to cpp macros for defining constants
#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512
const int window_width = WINDOW_WIDTH;
const int window_height = WINDOW_HEIGHT;

#define MESH_WIDTH 256
#define MESH_HEIGHT 256
//#define MESH_WIDTH 2048
//#define MESH_HEIGHT 2048

const int mesh_width = MESH_WIDTH;
const int mesh_height = MESH_HEIGHT;

// our simulation mesh
float v[MESH_WIDTH * MESH_HEIGHT];       // velocity
float u[MESH_WIDTH * MESH_HEIGHT];       // amplitude


// physical units for grid spacing, time
const float dx2 = 1.0;
const float dy2 = 1.0;
const float dt  = 0.01;

// wave propertiess
#define G_C 100.
#define G_K (6.28 / 367.)

const float g_c = G_C;                   // phase velocity
const float g_k = G_K;                   // wave number.. just some nice number
const float g_w = G_K * G_C;             // wave frequency

// image dimension (for interior boundary condition)
#define IMG_X 248  
#define IMG_Y 189
const int img_x = IMG_X;
const int img_y = IMG_Y;

float img[IMG_X * IMG_Y];


typedef struct cudaGraphicsResource cudaGraphicsResourceT;

// Vertex buffer
GLuint vbo;
cudaGraphicsResourceT* cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// Normals buffer
GLuint normBuffer;
struct cudaGraphicsResource* cuda_normBuffer_resource;

// Color buffer
GLuint colBuffer;
struct cudaGraphicsResource* cuda_colBuffer_resource;

// index buffer for storing the indices of triangle strips
GLuint indexBuffer;

float g_fAnim = 0.0;      // global time
int g_fImg = 1;           // flag indicating logo to be displayd

// we artificially highlight the logo in the center. Color amplitude
// is increased over fade_in timesteps

//float fade_in = 300.;  looks good, too!
float fade_in = 500.;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;


int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
void runCuda();
void cleanup();

// GL functionality
void initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, cudaGraphicsResourceT** cuda_res, int size, unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo);

void createMeshIndexBuffer(GLuint *id, int w, int h);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

const char *sSDKsample = "OpenACC-OpenGL";


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    printf("%s starting...\n", sSDKsample);

    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
void initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("OpenACC + OpenGL Interop");

    glutFullScreen();
    
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    glewInit();

    if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    //glDisable(GL_DEPTH_TEST);
    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_LIGHTING); 
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);

    // define material and surface properties
    GLfloat no_mat[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat mat_ambient[] = { 0.2, 0.7, 0.1, 1.0 };
    GLfloat mat_ambient_color[] = { 0.2, 0.8, 0.2, 1.0 };
  // GLfloat mat_diffuse[] = { 0.1, 0.5, 0.8, 1.0 };
    GLfloat mat_diffuse[] = { 0.1, 0.5, 0.1, 1.0 };
    GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat no_shininess[] = { 0.0 };
    GLfloat low_shininess[] = { 5.0 };
    GLfloat high_shininess[] = { 100.0 };
    GLfloat mat_emission[] = {0.3, 0.2, 0.2, 0.0};

    glTranslatef (1.25, 3.0, 0.0);
 //  glMaterialfv(GL_FRONT, GL_AMBIENT, no_mat);
    glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
 
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);
    glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);

    GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

}

////////////////////////////////////////////////////////////////////////////////
//! Read the logo file and store it in the img array. The image is stored
//! in pgm format. We assume that the img array is already correctly sized
//! and no error checking is performed.
////////////////////////////////////////////////////////////////////////////////
void readImg(){
FILE* f;
char buf[1024];
int x,y, c;

  printf("reading logo\n");
  f = fopen("nvlogo_gray.pgm", "rt");
  fscanf(f, "%s", buf); // header 
  printf("line %s\n", buf); 
  fscanf(f, "%d %d", &x, &y);
  printf("%d %d\n", x, y);
  fscanf(f, "%d", &c);
  printf("%d\n", c);
  for(int y=0; y<img_y; y++){
    for(int x=0; x<img_x; x++){
      fscanf(f, "%d", &c);
      img[y*img_x + x]= c;
    }
  }
  printf("logo read\n");
  fclose(f);
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
  initGL(&argc, argv);

  // register callbacks
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);

  // create VBO
  createVBO(&vbo, &cuda_vbo_resource, mesh_width*mesh_height*4*sizeof(float), 2);
  createVBO(&normBuffer, &cuda_normBuffer_resource, mesh_width*mesh_height*3*sizeof(float), 2);
  createVBO(&colBuffer, &cuda_colBuffer_resource, mesh_width*mesh_height*4*sizeof(float), 2);
  
  createMeshIndexBuffer(&indexBuffer, mesh_width, mesh_height);
   
  cudaGLSetGLDevice(0);

  for(int y=0; y<mesh_height; y++)
   for(int x=0; x<mesh_width; x++){ 
     u[(x+y*mesh_width)] = 0.0;
     v[(x+y*mesh_width)] = 0.0;
   }

  // load the logo
  readImg();
  printf("Image read, about to enter main loop\n");

  // From here on the simulation variables live on the GPU
#pragma acc data copy(u,v, img) 
{
  printf("after data region\n");
  glutMainLoop();
}
  atexit(cleanup);
  return ;
}


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, cudaGraphicsResourceT** cuda_res, int size, unsigned int vbo_res_flags)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(cuda_res, *vbo, vbo_res_flags);
}


////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo)
{
    // unregister this buffer object with CUDA
    cudaGraphicsUnregisterResource(cuda_vbo_resource);

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//!  create index buffer for rendering the mesh via triangle strip
////////////////////////////////////////////////////////////////////////////////
void createMeshIndexBuffer(GLuint *id, int pw, int ph)
{   
    int w = pw;
    int h = ph;
  
    int size = (2 * w * h - h) * sizeof(GLuint);
    // create index buffer
    glGenBuffersARB(1, id);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
    glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

    // fill with indices for rendering mesh as triangle strips
    GLuint *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

    if (!indices)
    { 
        return;
    }

    int x, y;
    
     for (y=0; y<h-1; )
    {
        for (x=0; x<w; x++)
        {
            *indices++ = y*w+x;         
            *indices++ = (y+1)*w+x; 
        }
        y++;
        
        for (x = (w-2); x>=0; x--)
        {
            *indices++ = (y+1)*w+(x+1);    
            *indices++ = y*w+x; 
        }     
        y++;
    }

    glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}


////////////////////////////////////////////////////////////////////////////////
//! Run the CUDA part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
    // map OpenGL buffer objects for writing from CUDA
    float *restrict dptr;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);

    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource);

    float *restrict dptr_norm;
    cudaGraphicsMapResources(1, &cuda_normBuffer_resource, 0);
    size_t num_bytes_norm;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr_norm, &num_bytes_norm, cuda_normBuffer_resource);

    float *restrict dptr_col;
    cudaGraphicsMapResources(1, &cuda_colBuffer_resource, 0);
    size_t num_bytes_col;
    cudaGraphicsResourceGetMappedPointer((void **)&dptr_col, &num_bytes_col, cuda_colBuffer_resource);

    // just for convenience
    float time = g_fAnim;

#pragma acc kernels deviceptr(dptr, dptr_norm, dptr_col) present(u,v, img)
  {
   
    if(time==0.0f){
      // clear out the simulation fields
      for(int y=0; y<mesh_height; y++){
        for(int x=0; x< mesh_width; x++){
         v[x+y*mesh_width] = 0.0;
         u[x+y*mesh_width] = 0.0;
        }
      }
      // clear out the triangle field
      for(int y=0; y<mesh_height; y++){
        for(int x=0; x<mesh_width; x++){
          float s = x/(float)mesh_width;
          float t = y/(float)mesh_height;
          dptr[(x+y*mesh_width)*4  ] = 2.f * s-1.f;  // x-coordinate
          dptr[(x+y*mesh_width)*4+1] = 2.f * t-1.f;  // y-coordinate
          dptr[(x+y*mesh_width)*4+2] = 0.0;          // z-coordinate
          dptr[(x+y*mesh_width)*4+3] = 1.0;          

        }
      } 
    }

    // amplitude update: u_new = u_old + dt * v
    for(int y=0; y<mesh_height; y++){
      for(int x=0; x<mesh_width; x++){
        u[(y*mesh_width+x)] += dt * v[y*mesh_width+x];
      } 
    } 

    // velocity update: v_new = v_old + dt * c * laplacin(u)
    for(int y=1; y<mesh_height-2; y++){
      for(int x=1; x<mesh_width-2; x++){
        v[y*mesh_width + x] += dt * g_c * (
             (u[(x + (y+1)*mesh_width)] 
       - 2. * u[(x +     y*mesh_width)] 
            + u[(x + (y-1)*mesh_width)]) / dx2 + 
             (u[(x+1 + y*mesh_width)] 
       - 2. * u[(x   + y*mesh_width)] 
            + u[(x-1 + y*mesh_width)]) / dy2);
     }
    }


    // set the boundary condition
#pragma acc loop independent
    for(int y=0; y<2; y++){
      for(int x=0; x<mesh_width/4; x++){
           u[(y*mesh_width+x)] = 
              cosf(g_w*time + g_k*x) * 
              sinf(g_w*time + g_k*y) ;
      }

    } 
    // set the boundary condition
#pragma acc loop independent
    for(int y=0; y<mesh_height/4; y++){
      for(int x=0; x<2; x++){
        u[(y*mesh_width+x)] = 
              sinf(g_w*time + g_k*x) * 
              cosf(g_w*time + g_k*y) ;
      }
   }


   // if desired, set interior boundary condition   
   if( g_fImg == 1){
       
   // we want to offset the image a bit
   int off_x = 30;
   int off_y = 50;

#pragma acc loop independent
    for(int yi=0; yi<img_y; yi++){
      for(int xi=0; xi<img_x; xi++){
        if(img[yi*img_x+xi] < 10) {
            u[(yi+off_y)*mesh_width+(xi+off_x)] = 0.;
            v[(yi+off_y)*mesh_width+(xi+off_x)] = (time < fade_in) ? sin(6.28*time/fade_in)*sin(6.28*time/fade_in) : 1.;
        }
      }
   }

   }

   // store amplitude in vbo for visualization
#pragma acc loop independent
    for(int y=0; y<mesh_height; y++){
      for(int x=0; x<mesh_width; x++){
        dptr[(y*mesh_width+x)*4+2] = u[(y*mesh_width+x)]; 
      } 
    }
    // compute triangle normals and store in normals buffer
#pragma acc loop independent
    for(int y=0; y<mesh_height-1; y++){
      for(int x=0; x<mesh_width-1; x++){
        dptr_norm[(y*mesh_width+x)*3+0] = u[(y*mesh_width+(x+1))]- u[(y*mesh_width+x)]; 
        dptr_norm[(y*mesh_width+x)*3+1] = u[((y+1)*mesh_width+x)]- u[(y*mesh_width+x)]; 
        dptr_norm[(y*mesh_width+x)*3+2] = 1.0; 
      } 
    }
    // compute triangle color and store in color buffer
#pragma acc loop independent
    for(int y=0; y<mesh_height; y++){
      for(int x=0; x<mesh_width; x++){
        dptr_col[(y*mesh_width+x)*4+0] = 0.1; 
        dptr_col[(y*mesh_width+x)*4+1] = (v[(y*mesh_width + x)] + 1.)/2.; 
        dptr_col[(y*mesh_width+x)*4+2] = (0.01*v[((y+1)*mesh_width+x)]+0.01)/2.;  
        dptr_col[(y*mesh_width+x)*4+3] = 1.0; 

      } 
    }
  }  

    // unmap buffer object
    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_normBuffer_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_colBuffer_resource, 0);
}



////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    GLfloat lightpos[] = {0., 0., 0., 1.};
    
    glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
   
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    
    glBindBuffer(GL_ARRAY_BUFFER, normBuffer);
    glNormalPointer(GL_FLOAT, 0, 0);
    glEnableClientState(GL_NORMAL_ARRAY);
    
    glBindBuffer(GL_ARRAY_BUFFER, colBuffer);
    glColorPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);

    // glPolygonMode(GL_FRONT_AND_BACK, wireFrame ? GL_LINE : GL_FILL);
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
    glDrawElements(GL_TRIANGLE_STRIP, 2*(mesh_width-1)*(mesh_height-1) + mesh_width, GL_UNSIGNED_INT, 0);
       
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
 
    glutSwapBuffers();

    g_fAnim += dt;
}

void timerEvent(int value)
{
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void cleanup()
{
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case (27) :
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
            break;
        case (' '):      // reset simulation
            g_fAnim = 0.0;
            break;

        case ('l'):      // turn on interior boundary condition
            g_fImg = 1;
            break;

        case ('n'):      // turn off interior boundary conditions
            g_fImg = 0;
 
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

