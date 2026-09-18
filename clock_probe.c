// What tick can two processes on this machine share? Measure the timer's
// resolution, the jitter of a timed wait, and the jitter of an edge crossing
// a pipe. The dial can be no finer than the jitter.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mach/mach_time.h>
static double ns(){return (double)mach_absolute_time()*1.0;}   // M-series: mach ticks are ns
int main(){
  mach_timebase_info_data_t tb; mach_timebase_info(&tb);
  printf("timer: mach_absolute_time, %u/%u ns per tick\n", tb.numer, tb.denom);
  // 1. resolution: smallest nonzero step
  double t0=ns(), t1; int k=0; do{t1=ns();k++;}while(t1==t0);
  printf("  resolution: %.0f ns (after %d reads)\n", (t1-t0)*tb.numer/tb.denom, k);
  // 2. jitter of a busy-wait for a target instant (what a sender does to fire at its tick)
  double worst=0, sum=0; int n=2000;
  for(int i=0;i<n;i++){ double target=ns()+2000; double now; do{now=ns();}while(now<target); double e=now-target; sum+=e; if(e>worst)worst=e; }
  printf("  busy-wait to an instant: mean late %.0f ns, worst %.0f ns\n", sum/n, worst);
  // 3. jitter of an edge across a pipe: parent writes 1 byte at a scheduled instant, child timestamps arrival
  int p[2]; pipe(p); pid_t pid=fork();
  if(pid==0){ close(p[1]); unsigned char b; double prev=0, worstj=0, sumd=0, mind=1e18, maxd=0; int m=0;
    while(read(p[0],&b,1)==1){ double t=ns(); if(prev){ double d=t-prev; sumd+=d; if(d<mind)mind=d; if(d>maxd)maxd=d; m++; } prev=t; }
    printf("  edges across a pipe, sent every 50 us: arrival spacing mean %.1f us, min %.1f, max %.1f us  (n=%d)\n", sumd/m/1e3, mind/1e3, maxd/1e3, m);
    _exit(0); }
  close(p[0]); unsigned char b=1; double next=ns()+100000;
  for(int i=0;i<2000;i++){ double now; do{now=ns();}while(now<next); write(p[1],&b,1); next+=50000; }
  close(p[1]); int st; wait(&st); return 0; }
