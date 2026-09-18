// EXACT AT SPEED: nested, gap-coded, protected. Core to core, fast tick.
//   three nests, radix 16, each digit a gap of (digit+1)*STEP ticks
//   STEP ticks per digit step: a slip smaller than STEP/2 rounds back (corrects the common +-1)
//   a check digit (sum of digits mod 16) catches a big slip
//   every weight sent twice; on disagreement the copy whose check passes wins
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define N 768
#define HALF 2047
#define R 16
#define ND 3
int main(int argc,char**argv){
  double tick=argc>1?atof(argv[1]):42;
  int STEP=argc>2?atoi(argv[2]):3;
  int copies=argc>3?atoi(argv[3]):2;
  int per = (ND+1)*copies;                                   // edges per weight: digits + check, x copies
  volatile long*flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile double*ts=mmap(0,(N*per+2)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*dir=mmap(0,(N*per+2)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  srand(7); float w[N],x[N]; float cyc=0;
  for(int i=0;i<N;i++){ double u=rand()/(double)RAND_MAX,v=rand()/(double)RAND_MAX; w[i]=(float)(sqrt(-2*log(u+1e-12))*cos(6.2831853*v))*0.02f;
                        x[i]=(float)rand()/RAND_MAX-0.5f; if(fabsf(w[i])>cyc)cyc=fabsf(w[i]); }
  int q[N]; for(int i=0;i<N;i++) q[i]=(int)lroundf(w[i]/cyc*HALF);
  pid_t pid=fork();
  if(pid==0){ long seen=0; int k=0; while(k<N*per+1){ long f; while((f=*flag)==seen); dir[k]=f>seen; seen=f; ts[k++]=ns(); } _exit(0); }
  usleep(20000); *flag=0; double t0=ns()+200000; while(ns()<t0); long e=1; *flag=e;
  double at=t0;
  for(int i=0;i<N;i++){ int m=abs(q[i]); int d[ND+1]; int s=0;
    for(int j=0;j<ND;j++){ d[j]=m%R; m/=R; s+=d[j]; } d[ND]=s%R;          // digits low->high, then check
    for(int c=0;c<copies;c++) for(int j=0;j<=ND;j++){
      at += (double)(d[j]+1)*STEP*tick; while(ns()<at); *flag = (q[i]>=0)? ++e : --e; } }
  double tend=ns(); int st; wait(&st);
  double acc=0,ref=0; int bad=0,disagree=0,fixed=0,raw_bad=0;
  int k=1;
  for(int i=0;i<N;i++){ int val[4]; int ok[4]; int sgn=0;
    for(int c=0;c<copies;c++){ int d[ND+1];
      for(int j=0;j<=ND;j++){ double gap=(ts[k]-ts[k-1])/tick; d[j]=(int)floor(gap/STEP+0.5)-1; if(d[j]<0)d[j]=0; if(d[j]>=R)d[j]=R-1; sgn=dir[k]?1:-1; k++; }
      int s=0,v=0; for(int j=ND-1;j>=0;j--){ v=v*R+d[j]; s+=d[j]; }
      val[c]=v; ok[c]=(s%R==d[ND]); }
    int v=val[0]; int rawwrong = (val[0]!=abs(q[i]));
    if(copies>1){ if(val[0]!=val[1]){ disagree++; if(!ok[0]&&ok[1]) v=val[1]; if(ok[0]&&!ok[1]) v=val[0]; if(v==abs(q[i]) && rawwrong) fixed++; } }
    if(rawwrong) raw_bad++;
    int got=sgn*v; if(got!=q[i]) bad++;
    acc+=(double)got/HALF*x[i]; ref+=(double)q[i]/HALF*x[i]; }
  double secs=(tend-t0)/1e9;
  printf("tick %3.0f ns  step %d  copies %d : before correction %d wrong; copies disagreed on %d, check fixed %d; AFTER: %d wrong of %d   dot rel err %.1e   %.0f weights/s on one line\n",
    tick, STEP, copies, raw_bad, disagree, fixed, bad, N, fabs(acc-ref)/fabs(ref), N/secs);
  return 0; }
