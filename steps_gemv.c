// The door test: rows stored as SORTED STEPS, 4 bits each (0..14; 15 = escape,
// next 12 bits hold a big step). The kernel decodes and accumulates in one
// pass, batch 1, all threads. Order is wiring: each row's input permutation is
// stored once (int16) and never crosses per token... except it does here,
// because on a CPU the gather x[perm[i]] is a read too. Both costs measured.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define NIN 768
#define ROWS 110592          // GPT-2 linear layers: 84.9M weights / 768
typedef struct { const uint8_t*code; const uint32_t*off; const int16_t*perm; const float*x; float*y; size_t r0,r1; int use_perm; } job;
static void* run(void*p){ job*J=p; const float*x=J->x;
  for(size_t r=J->r0;r<J->r1;r++){ const uint8_t*c=J->code+J->off[r]; const int16_t*pm=J->perm+r*NIN;
    int v=-2047; float acc=0; size_t bit=0;
    for(int i=0;i<NIN;i++){ uint32_t nib=(c[bit>>3]>>(bit&4))&15; bit+=4;
      if(nib==15){ uint32_t big=(c[bit>>3]>>(bit&4))&15; bit+=4; big|=((c[bit>>3]>>(bit&4))&15)<<4; bit+=4; big|=((c[bit>>3]>>(bit&4))&15)<<8; bit+=4; v+=big; } else v+=nib;
      acc += v * x[J->use_perm? pm[i] : i]; }
    J->y[r]=acc/2047.0f; }
  return 0; }
int main(){
  // build sorted-step rows with GPT-2's step statistics: gaussian weights, per-row scaled
  size_t nw=(size_t)ROWS*NIN; uint8_t*code=malloc(nw); uint32_t*off=malloc((ROWS+1)*4); int16_t*perm=malloc(nw*2);
  float*x=malloc(NIN*4); for(int i=0;i<NIN;i++) x[i]=(float)rand()/RAND_MAX-0.5f; float*y=malloc(ROWS*4);
  srand(1); size_t bit=0; double bits_total=0; int*q=malloc(NIN*4); int*ord=malloc(NIN*4);
  for(size_t r=0;r<ROWS;r++){ off[r]=bit>>3; float mx=0; float w[NIN];
    for(int i=0;i<NIN;i++){ double u=rand()/(double)RAND_MAX,v2=rand()/(double)RAND_MAX; w[i]=(float)(sqrt(-2*log(u+1e-12))*cos(6.2831853*v2)); if(fabsf(w[i])>mx)mx=fabsf(w[i]); }
    for(int i=0;i<NIN;i++){ q[i]=(int)lroundf(w[i]/mx*2047); ord[i]=i; }
    for(int i=1;i<NIN;i++){ int k=ord[i],j=i-1; while(j>=0&&q[ord[j]]>q[k]){ord[j+1]=ord[j];j--;} ord[j+1]=k; }
    int prev=-2047; for(int i=0;i<NIN;i++){ int s=q[ord[i]]-prev; prev=q[ord[i]]; perm[r*NIN+i]=(int16_t)ord[i];
      if(s<15){ code[bit>>3] = (bit&4)? (code[bit>>3]&0x0F)|(s<<4) : s; bit+=4; }
      else { uint32_t nibs[4]={15,(uint32_t)s&15,((uint32_t)s>>4)&15,((uint32_t)s>>8)&15}; for(int n=0;n<4;n++){ code[bit>>3]=(bit&4)?(code[bit>>3]&0x0F)|(nibs[n]<<4):nibs[n]; bit+=4; } } } }
  off[ROWS]=bit>>3; bits_total=(double)bit;
  printf("GPT-2-sized: %zu weights as sorted steps = %.1f MB (%.2f bits/weight)  vs 12-bit %.1f MB  vs fp32 %.1f MB\n", nw, bit/8/1e6, bits_total/nw, nw*1.5/1e6, nw*4/1e6);
  int T=10;
  for(int up=0;up<2;up++){ double best=1e9;
    for(int rep=0;rep<5;rep++){ pthread_t th[16]; job J[16]; size_t per=ROWS/T; double t0=now();
      for(int t=0;t<T;t++){ J[t]=(job){code,off,perm,x,y,(size_t)t*per,(size_t)(t+1)*per,up}; pthread_create(&th[t],0,run,&J[t]); }
      for(int t=0;t<T;t++) pthread_join(th[t],0); double d=now()-t0; if(d<best)best=d; }
    double mb=bit/8/1e6+(up? nw*2/1e6:0);
    printf("  decode+accumulate, %s: %6.2f ms/token  (%.0f MB moved)  -> %5.0f tok/s ceiling\n", up?"gathering x by the stored order (perm read per token)":"in stored order (perm free)", best*1e3, mb, 1/best); }
  return 0; }
