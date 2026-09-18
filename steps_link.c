// How many distinct edge steps survive the link? Each edge is a counter jump
// of +-1 .. +-K, chosen at random; the receiver reads the jump. Timing as in
// sorted_link (guarded). Wrong = jump read differently from jump sent.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define N 2000
int main(int argc,char**argv){
  double tick=argc>1?atof(argv[1]):200; int K=argc>2?atoi(argv[2]):2; int guard=argc>3?atoi(argv[3]):5;
  volatile long*flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*got=mmap(0,(N+16)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  srand(3); long sent[N]; for(int i=0;i<N;i++){ int m=1+rand()%K; sent[i]=(rand()&1)?m:-m; }
  pid_t pid=fork();
  if(pid==0){ alarm(5); long seen=0; int k=0; while(k<N+10){ long f; while((f=*flag)==seen); got[k++]=f-seen; seen=f; } _exit(0); }
  usleep(20000); *flag=0; long e=0; double t0=ns()+200000;
  for(int k=0;k<10;k++){ while(ns()<t0+k*100*tick); *flag=++e; }
  double at=t0+900*tick;
  for(int i=0;i<N;i++){ at+=guard*tick; while(ns()<at); e+=sent[i]; *flag=e; }
  int st; wait(&st);
  int wrong=0, merged=0; for(int i=0;i<N;i++){ if(got[i+10]!=sent[i]) wrong++; }
  printf("tick %4.0f ns  steps +-1..+-%-4d (%d symbols, %.1f bits/edge)  guard %d ticks: %d wrong of %d\n", tick, K, 2*K, (K==1?1.0:(K==2?2.0:(K==8?4.0:(K==128?8.0:(K==2048?12.0:0.0))))), guard, wrong, N);
  return 0; }
