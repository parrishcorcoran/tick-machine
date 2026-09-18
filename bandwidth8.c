// The multiply done IN 8 bits: SDOT (int8 x int8 -> int32), 4 weights per lane
// per instruction. No conversion to float per weight. Does 8-bit ride the wall?
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define NW 123532032ULL
#define NIN 768
typedef struct { const int8_t*W; size_t rows; const int8_t*x; float*y; int four; } job;
static void* run(void*p){ job*J=p; const int8_t*x=J->x;
  for(size_t r=0;r<J->rows;r++){ int32x4_t acc=vdupq_n_s32(0);
    if(!J->four){ const int8_t*w=J->W+r*NIN;
      for(int i=0;i<NIN;i+=16) acc=vdotq_s32(acc,vld1q_s8(w+i),vld1q_s8(x+i)); }
    else { const uint8_t*w=(const uint8_t*)J->W+r*(NIN/2);       // 4-bit pairs, unpack to int8 then SDOT
      for(int i=0;i<NIN;i+=32){ uint8x16_t b=vld1q_u8(w+i/2);
        int8x16_t lo=vreinterpretq_s8_u8(vandq_u8(b,vdupq_n_u8(15))), hi=vreinterpretq_s8_u8(vshrq_n_u8(b,4));
        acc=vdotq_s32(acc,vzip1q_s8(lo,hi),vld1q_s8(x+i)); acc=vdotq_s32(acc,vzip2q_s8(lo,hi),vld1q_s8(x+i+16)); } }
    J->y[r]=(float)vaddvq_s32(acc); }
  return 0; }
int main(){ size_t rows=NW/NIN; int8_t*W=malloc(NW); memset(W,3,NW); int8_t x[NIN]; for(int i=0;i<NIN;i++)x[i]=(int8_t)(i%7-3); float*y=malloc(rows*4);
  for(int four=0;four<2;four++) for(int T=4;T<=10;T+=6){ double best=1e9;
    for(int rep=0;rep<5;rep++){ pthread_t th[16]; job J[16]; size_t per=rows/T; double t0=now();
      for(int t=0;t<T;t++){ J[t].W=W+(size_t)t*per*(four?NIN/2:NIN); J[t].rows=per; J[t].x=x; J[t].y=y+t*per; J[t].four=four; pthread_create(&th[t],0,run,&J[t]); }
      for(int t=0;t<T;t++) pthread_join(th[t],0); double d=now()-t0; if(d<best)best=d; }
    double mb=NW*(four?0.5:1.0)/1e6; printf("  %d-bit SDOT, %2d threads: %6.1f MB/token  %5.2f ms  %5.1f GB/s  -> %5.0f tok/s ceiling\n", four?4:8, T, mb, best*1e3, mb/1e3/best, 1/best); }
  return 0; }
