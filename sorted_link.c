// #2 SORTED STEPS ON THE REAL LINK: each column sorted once (the wiring); a weight is the step up from the previous one.
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
#define GUARD_T 4
#define D 64      // the small dial: timing carries step % D, the edge symbol carries step / D   // ticks after every edge so no two edges land inside the trip jitter
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define N 768
#define HALF 2047
int main(int argc,char**argv){
  double tick=argc>1?atof(argv[1]):200;
  int mode=argc>2?atoi(argv[2]):1;                 // 1 = gap coding, 0 = flat dial
  volatile long*flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*level=flag+1;                     // the resting sign of the line: +0 / -0
  volatile long*lv_seen=mmap(0,(N+12)*8*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);  // receiver's samples of the level, per idle tick
  volatile double*ts=mmap(0,(N+12)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*dir=mmap(0,(N+12)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  // real GPT-2-like weights: gaussian, per-column scaled so the largest is the dial's half
  srand(7); float w[N],x[N]; float cyc=0;
  for(int i=0;i<N;i++){ double u=rand()/(double)RAND_MAX, v=rand()/(double)RAND_MAX; w[i]=(float)(sqrt(-2*log(u+1e-12))*cos(6.2831853*v))*0.02f;
                        x[i]=(float)rand()/RAND_MAX-0.5f; if(fabsf(w[i])>cyc)cyc=fabsf(w[i]); }
  int q[N]; for(int i=0;i<N;i++) q[i]=(int)lroundf(w[i]/cyc*HALF);              // signed angle
  int order[N]; for(int i=0;i<N;i++) order[i]=i;                                    // the wiring: sorted once, stored at both ends
  for(int i=1;i<N;i++){ int k=order[i], j=i-1; while(j>=0 && q[order[j]]>q[k]){ order[j+1]=order[j]; j--; } order[j+1]=k; }
  int lapbits[N]; for(int i=0;i<N;i++) lapbits[i]=rand()&15;   // the 4-bit lap counter to carry in the idle ticks
  int step[N]; step[0]=q[order[0]]+HALF; for(int i=1;i<N;i++) step[i]=q[order[i]]-q[order[i-1]];   // first step from -HALF
  pid_t pid=fork();
  if(pid==0){ alarm(5); long seen=0; int k=0;
    while(k<N+10){ long f; while((f=*flag)==seen); dir[k]=f-seen; seen=f; ts[k++]=ns(); } _exit(0); }
  usleep(20000); *flag=0; long lv=0; double t0=ns()+200000;
  for(int k=0;k<10;k++){ while(ns()<t0+k*100*tick); *flag=++lv; }                  // ten warm-up pilots; the last is the reference
  double at=t0+900*tick; long e=lv;
  for(int i=0;i<N;i++){
    if(mode){ at += (step[i]%D+1+GUARD_T)*tick;  while(ns()<at); e += 1 + step[i]/D; *flag = e; }   // timing: step mod D; symbol: 1 + step/D
    else    { double turn=t0+(double)(i+1)*4096*tick; int a=((q[i]%4096)+4096)%4096; while(ns()<turn+a*tick); *flag=++e; }
  }
  double tend=ns(); int st; wait(&st);
  // realign: edge k carries counter delta dir[k]; its true index is the running counter
  static double tsa[N+12]; static int have[N+12]; static long sym[N+12]; { long c=0; int idx=0; for(int k=0;k<N+10;k++){ if(dir[k]==0) continue; if(k<10){ c+=dir[k]; tsa[c-1]=ts[k]; have[c-1]=1; idx=(int)c; } else { tsa[idx]=ts[k]; sym[idx]=dir[k]; have[idx]=1; idx++; } } }
  int missed=0; for(int k=0;k<N+10;k++) if(!have[k]) missed++;
  double acc=0, ref=0; int bad=0, maxoff=0;
  for(int i=0;i<N;i++){ int got;
    if(mode){ if(!have[i+10]){ got=q[order[i]]; bad++; continue; } static long hisum; if(i==0) hisum=0; hisum += ((long)sym[i+10]-1)*D;   /* laps: exact, from the symbols */
              double tt=(tsa[i+10]-tsa[9])/tick; long losum=(long)floor(tt+0.5)-(long)(i+1)*(1+GUARD_T);   /* timing: from absolute time, no accumulation */
              got=(int)(hisum+losum-HALF); }   // nominal tick (same clock both ends), from the last warm pilot
    else    { double t=ts[i+1]-ts[0]-(double)(i+1)*4096*tick; int a=(int)floor(t/tick+0.5); a=((a%4096)+4096)%4096; got=(a>2048)?a-4096:a; }
    int idx = mode? order[i] : i;
    int off=abs(got-q[idx]); if(off){bad++; if(off>maxoff)maxoff=off;}
    acc+=(double)got/HALF*x[idx]; ref+=(double)q[idx]/HALF*x[idx]; }
  if(0){ for(int i=0;i<6;i++){ double tk=tick; double tt=(ts[i+10]-ts[9])/tk; printf("    i=%d sent q=%d step=%d  arrival %.2f ticks  expect %d  got %d\n", i, q[order[i]], step[i], tt, q[order[i]]+HALF+(i+1)*(1+GUARD_T), (int)floor(tt+0.5)-(i+1)*(1+GUARD_T)-HALF);} }
  double secs=(tend-t0)/1e9;
  printf("missed edges %d | tick %4.0f ns  %-10s: %d wrong of %d (worst off by %d ticks)   dot %.5f vs %.5f (rel err %.1e)   %.0f weights/s on one line\n",
    missed, tick, mode?"sorted, dial 64 + lap symbol":"flat 4096", bad, N, maxoff, acc, ref, fabs(acc-ref)/fabs(ref), N/secs);
  return 0; }
