// GAP CODING ON THE REAL LINK. The dial is a counter that resets at every edge.
// A weight is the gap since the previous edge (its |angle|); its sign is the
// edge's direction (the shared flag goes UP for positive, DOWN for negative).
// No lap is waited for. The receiver: read the counter, note the direction,
// accumulate, reset. Compared against the flat 4096 dial on the same weights.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
#include <signal.h>
#define GUARD_T 4   // ticks after every edge so no two edges land inside the trip jitter
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define N 768
#define HALF 2047
int main(int argc,char**argv){
  double tick=argc>1?atof(argv[1]):200;
  int mode=argc>2?atoi(argv[2]):1;                 // 1 = gap coding, 0 = flat dial
  volatile long*flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile double*ts=mmap(0,(N+2)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*dir=mmap(0,(N+2)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  // real GPT-2-like weights: gaussian, per-column scaled so the largest is the dial's half
  srand(7); float w[N],x[N]; float cyc=0;
  for(int i=0;i<N;i++){ double u=rand()/(double)RAND_MAX, v=rand()/(double)RAND_MAX; w[i]=(float)(sqrt(-2*log(u+1e-12))*cos(6.2831853*v))*0.02f;
                        x[i]=(float)rand()/RAND_MAX-0.5f; if(fabsf(w[i])>cyc)cyc=fabsf(w[i]); }
  int q[N]; for(int i=0;i<N;i++) q[i]=(int)lroundf(w[i]/cyc*HALF);              // signed angle
  pid_t pid=fork();
  if(pid==0){ alarm(5); long seen=0; int k=0;
    while(k<N+1){ long f; while((f=*flag)==seen); dir[k]=f>seen; seen=f; ts[k++]=ns(); } _exit(0); }
  usleep(20000); *flag=0; long lv=0; double t0=ns()+200000;
  while(ns()<t0); *flag=++lv;                                                     // pilot edge at t0
  double at=t0; long e=lv;
  for(int i=0;i<N;i++){
    if(mode){ at += (abs(q[i])+1+GUARD_T)*tick;  while(ns()<at); *flag = (q[i]>=0) ? ++e : --e; }   // gap = |angle|, direction = sign
    else    { double turn=t0+(double)(i+1)*4096*tick; int a=((q[i]%4096)+4096)%4096; while(ns()<turn+a*tick); *flag=++e; }
  }
  double tend=ns(); int st; wait(&st);
  double acc=0, ref=0; int bad=0, maxoff=0;
  for(int i=0;i<N;i++){ int got;
    if(mode){ double gap=(ts[i+1]-ts[i])/tick; int mag=(int)floor(gap+0.5)-1-GUARD_T; got = dir[i+1]? mag : -mag; }
    else    { double t=ts[i+1]-ts[0]-(double)(i+1)*4096*tick; int a=(int)floor(t/tick+0.5); a=((a%4096)+4096)%4096; got=(a>2048)?a-4096:a; }
    int off=abs(got-q[i]); if(off){bad++; if(off>maxoff)maxoff=off;}
    acc+=(double)got/HALF*x[i]; ref+=(double)q[i]/HALF*x[i]; }
  double secs=(tend-t0)/1e9;
  printf("tick %4.0f ns  %-10s: %d wrong of %d (worst off by %d ticks)   dot %.5f vs %.5f (rel err %.1e)   %.0f weights/s on one line\n",
    tick, mode?"gap+sign":"flat 4096", bad, N, maxoff, acc, ref, fabs(acc-ref)/fabs(ref), N/secs);
  return 0; }
