// What this machine can move, and what a batch-1 pass over GPT-2's weights
// costs at each width. All threads. The ceiling for single-stream tokens/s.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <arm_neon.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define NW 123532032ULL              // GPT-2 weights incl. embeddings/head
#define NIN 768
typedef struct { const void*W; size_t rows; int bits; const float*x; float*y; } job;
static void* run(void*p){ job*J=p; const float*x=J->x;
  for(size_t r=0;r<J->rows;r++){ float32x4_t acc=vdupq_n_f32(0);
    if(J->bits==32){ const float*w=(const float*)J->W+r*NIN; for(int i=0;i<NIN;i+=4) acc=vmlaq_f32(acc,vld1q_f32(w+i),vld1q_f32(x+i)); }
    else if(J->bits==16){ const int16_t*w=(const int16_t*)J->W+r*NIN; for(int i=0;i<NIN;i+=4){ int16x4_t h=vld1_s16(w+i); acc=vmlaq_f32(acc,vcvtq_f32_s32(vmovl_s16(h)),vld1q_f32(x+i)); } }
    else if(J->bits==8){ const int8_t*w=(const int8_t*)J->W+r*NIN; for(int i=0;i<NIN;i+=8){ int8x8_t b=vld1_s8(w+i); int16x8_t h=vmovl_s8(b);
        acc=vmlaq_f32(acc,vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),vld1q_f32(x+i)); acc=vmlaq_f32(acc,vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))),vld1q_f32(x+i+4)); } }
    else { const uint8_t*w=(const uint8_t*)J->W+r*(NIN/2); for(int i=0;i<NIN;i+=8){ uint8x8_t b=vld1_u8(w+i/2); // 4-bit pairs
        int8x8_t lo=vreinterpret_s8_u8(vand_u8(b,vdup_n_u8(15))), hi=vreinterpret_s8_u8(vshr_n_u8(b,4));
        int16x8_t h=vmovl_s8(vzip1_s8(lo,hi));
        acc=vmlaq_f32(acc,vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))),vld1q_f32(x+i)); acc=vmlaq_f32(acc,vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))),vld1q_f32(x+i+4)); } }
    J->y[r]=vaddvq_f32(acc); }
  return 0; }
int main(){ int T=8; size_t rows=NW/NIN; size_t bytes32=NW*4;
  void*W=malloc(bytes32); memset(W,1,bytes32); float*x=malloc(NIN*4); for(int i=0;i<NIN;i++)x[i]=0.001f*i; float*y=malloc(rows*4);
  int widths[4]={32,16,8,4}; printf("GPT-2: %llu weights. batch-1 pass = read every weight once.\n", NW);
  for(int k=0;k<4;k++){ int bits=widths[k]; double best=1e9;
    for(int rep=0;rep<5;rep++){ pthread_t th[16]; job J[16]; size_t per=rows/T; double t0=now();
      for(int t=0;t<T;t++){ J[t].W=(char*)W+(size_t)t*per*NIN*bits/8; J[t].rows=per; J[t].bits=bits; J[t].x=x; J[t].y=y+t*per; pthread_create(&th[t],0,run,&J[t]); }
      for(int t=0;t<T;t++) pthread_join(th[t],0); double d=now()-t0; if(d<best)best=d; }
    double mb=NW*bits/8.0/1e6; printf("  %2d-bit weights: %7.1f MB/token  %6.2f ms/token  %6.1f GB/s  -> %6.0f tok/s ceiling\n", bits, mb, best*1e3, mb/1e3/best, 1/best); }
  return 0; }
