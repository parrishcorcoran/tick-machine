// Same dot product, three ways, each fast enough to hit the memory wall:
//   fp32 weights            4 bytes/weight
//   int16 ticks, signed     2 bytes/weight  (the dial table is linear: value = tick/half)
//   int8 ticks, signed      1 byte/weight   (a 256-dial)
// Batch 1 GEMV over enough layers that the weights cannot sit in cache.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define NIN 768
#define NOUT 3072
static void gemv_f32(const float*W,const float*x,float*y){
    for(int j=0;j<NOUT;j++){const float*w=W+(size_t)j*NIN;float s0=0,s1=0,s2=0,s3=0;
        for(int i=0;i<NIN;i+=4){s0+=w[i]*x[i];s1+=w[i+1]*x[i+1];s2+=w[i+2]*x[i+2];s3+=w[i+3]*x[i+3];}
        y[j]=s0+s1+s2+s3;}
}
static void gemv_i16(const int16_t*W,const float*x,float*y,float scale){
    for(int j=0;j<NOUT;j++){const int16_t*w=W+(size_t)j*NIN;float s0=0,s1=0,s2=0,s3=0;
        for(int i=0;i<NIN;i+=4){s0+=(float)w[i]*x[i];s1+=(float)w[i+1]*x[i+1];s2+=(float)w[i+2]*x[i+2];s3+=(float)w[i+3]*x[i+3];}
        y[j]=(s0+s1+s2+s3)*scale;}
}
static void gemv_i8(const int8_t*W,const float*x,float*y,float scale){
    for(int j=0;j<NOUT;j++){const int8_t*w=W+(size_t)j*NIN;float s0=0,s1=0,s2=0,s3=0;
        for(int i=0;i<NIN;i+=4){s0+=(float)w[i]*x[i];s1+=(float)w[i+1]*x[i+1];s2+=(float)w[i+2]*x[i+2];s3+=(float)w[i+3]*x[i+3];}
        y[j]=(s0+s1+s2+s3)*scale;}
}
int main(int argc,char**argv){
    int layers=argc>1?atoi(argv[1]):48; size_t n=(size_t)NIN*NOUT*layers;
    float*W=malloc(n*4);int16_t*T=malloc(n*2);int8_t*U=malloc(n);float x[NIN],y[NOUT];
    int half=2047; srand(1);
    for(size_t i=0;i<n;i++){int q=rand()%(2*half+1)-half;T[i]=(int16_t)q;W[i]=(float)q/half;U[i]=(int8_t)(q/16);}
    for(int i=0;i<NIN;i++)x[i]=(float)rand()/(float)RAND_MAX-0.5f;
    double bf=1e9,bt=1e9,bu=1e9;volatile float sink=0;
    for(int r=0;r<6;r++){
        double t0=now();for(int l=0;l<layers;l++){gemv_f32(W+(size_t)l*NIN*NOUT,x,y);sink+=y[0];}double d=(now()-t0)/layers;if(d<bf)bf=d;
        t0=now();for(int l=0;l<layers;l++){gemv_i16(T+(size_t)l*NIN*NOUT,x,y,1.0f/half);sink+=y[0];}d=(now()-t0)/layers;if(d<bt)bt=d;
        t0=now();for(int l=0;l<layers;l++){gemv_i8(U+(size_t)l*NIN*NOUT,x,y,16.0f/half);sink+=y[0];}d=(now()-t0)/layers;if(d<bu)bu=d;
    }
    double b=(double)NIN*NOUT;
    printf("%d layers, %.0f MB of fp32 weights (cache is ~16 MB), 768x3072 GEMV, batch 1, one core\n",layers,n*4/1e6);
    printf("  fp32  4 B/weight : %7.3f ms/layer  %6.1f GB/s\n",bf*1e3,b*4/bf/1e9);
    printf("  int16 2 B/weight : %7.3f ms/layer  %6.1f GB/s   %.2fx faster than fp32\n",bt*1e3,b*2/bt/1e9,bf/bt);
    printf("  int8  1 B/weight : %7.3f ms/layer  %6.1f GB/s   %.2fx faster than fp32\n",bu*1e3,b*1/bu/1e9,bf/bu);
    return 0;
}
