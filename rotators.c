// ROTATORS WITH THEIR OWN NOTES. N threads, each laps its own slice of int16 notes
// forever (acc += note * x), slice sized to live in the core's own cache (resident)
// or sized so it must come from RAM. Reports notes/s per rotator and in total.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
typedef struct { int16_t*w; long n; long laps; long sink; } job;
void* rot(void*a){ job*j=a; long s=0; for(long l=0;l<j->laps;l++){ int16_t x=(int16_t)(l%7+1); int32_t acc[8]={0}; for(long i=0;i+8<=j->n;i+=8) for(int k=0;k<8;k++) acc[k]+=(int32_t)j->w[i+k]*x; for(int k=0;k<8;k++) s+=acc[k]; } j->sink=s; return 0; }   /* 32-bit lanes: lets the adder vectorise */
int main(int argc,char**argv){
  long bytes=atol(argv[1]); int N=atoi(argv[2]); long n=bytes/2; long laps = (long)(2e9/bytes); if(laps<3) laps=3;
  job*J=calloc(N,sizeof(job)); pthread_t*T=malloc(N*sizeof(pthread_t));
  for(int t=0;t<N;t++){ J[t].w=malloc(bytes); for(long i=0;i<n;i++) J[t].w[i]=(int16_t)(i*7+t); J[t].n=n; J[t].laps=laps; }
  double t0=ns(); for(int t=0;t<N;t++) pthread_create(&T[t],0,rot,&J[t]); for(int t=0;t<N;t++) pthread_join(T[t],0); double dt=(ns()-t0)/1e9;
  double per=(double)n*laps/dt;
  printf("%2d rotators x %6.2f MB own notes: %6.1f G notes/s each, %7.1f G notes/s total (%5.0f GB/s)  -> GPT-2 (124M notes) laps at %5.0f /s if ALL notes were held this way\n",
    N, bytes/1e6, per/1e9, per*N/1e9, per*N*2/1e9, per*N/124e6);
  return 0; }
