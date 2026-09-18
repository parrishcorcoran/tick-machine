// #7 residue dials that correct themselves, and #10 three-direction edges.
// Each weight -> remainders on three coprime dials (61, 64, 67); each sent
// gap-coded (remainder+1 ticks). Any two remainders give the value (CRT);
// three pairs vote, so one slipped remainder is outvoted. Sign rides on the
// edge direction. --tri: the third direction (+2) marks a lap flag for free.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
#include <signal.h>
#define GUARD_T 6   // ticks after every edge so no two edges land inside the trip jitter
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define N 768
#define HALF 2047
static const int M[3]={61,64,67};
static int crt2(int a,int ma,int b,int mb){ for(int x=a;x<ma*mb;x+=ma) if(x%mb==b) return x; return -1; }
int main(int argc,char**argv){
  double tick=argc>1?atof(argv[1]):42; int tri=argc>2?atoi(argv[2]):0;
  volatile long*flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile double*ts=mmap(0,(N*3+2)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*dl=mmap(0,(N*3+2)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  srand(7); float w[N],x[N],cyc=0; int lap[N];
  for(int i=0;i<N;i++){ double u=rand()/(double)RAND_MAX,v=rand()/(double)RAND_MAX; w[i]=(float)(sqrt(-2*log(u+1e-12))*cos(6.2831853*v))*0.02f;
                        x[i]=(float)rand()/RAND_MAX-0.5f; if(fabsf(w[i])>cyc)cyc=fabsf(w[i]); lap[i]=(rand()%16==0); }
  int q[N]; for(int i=0;i<N;i++) q[i]=(int)lroundf(w[i]/cyc*HALF);
  pid_t pid=fork();
  if(pid==0){ alarm(5); long seen=0; int k=0; while(k<N*3+1){ long f; while((f=*flag)==seen); dl[k]=f-seen; seen=f; ts[k++]=ns(); } _exit(0); }
  usleep(20000); *flag=0; double t0=ns()+200000; while(ns()<t0); long e=1; *flag=e; double at=t0;
  for(int i=0;i<N;i++){ int a=abs(q[i]);
    for(int j=0;j<3;j++){ at+=(double)(a%M[j]+1+GUARD_T)*tick; while(ns()<at);
      long step = (q[i]>=0)? +1 : -1; if(tri && lap[i] && j==0) step*=2;   // lap flag: a double step, sign kept   // third direction: lap flag on the first edge
      e+=step; *flag=e; } }
  double tend=ns(); int st; wait(&st);
  int wrong=0, raw_wrong=0, lapwrong=0; double acc=0,ref=0; int k=1;
  for(int i=0;i<N;i++){ int r[3]; int sgn=0; int lapf=0;
    for(int j=0;j<3;j++){ double gap=(ts[k]-ts[k-1])/tick; r[j]=(int)floor(gap+0.5)-1-GUARD_T; if(r[j]<0)r[j]=0; if(r[j]>=M[j])r[j]=M[j]-1;
      if(dl[k]==2||dl[k]==-2) lapf=1; sgn=dl[k]>0?1:-1; k++; }
    int v01=crt2(r[0],M[0],r[1],M[1]), v02=crt2(r[0],M[0],r[2],M[2]), v12=crt2(r[1],M[1],r[2],M[2]);
    int v; if(v01==v02||v01==v12) v=v01; else if(v02==v12) v=v02; else { v=v01; int c=0; if(v01<=HALF){v=v01;c++;} if(v02<=HALF){v=v02;c++;} if(v12<=HALF){v=v12;c++;} }   // agree, else in range
    if(v>HALF) v=HALF;
    if(v01!=abs(q[i])) raw_wrong++;
    if(sgn*v!=q[i]) wrong++;
    if(tri && lapf!=lap[i]) lapwrong++;
    acc+=(double)(sgn*v)/HALF*x[i]; ref+=(double)q[i]/HALF*x[i]; }
  double secs=(tend-t0)/1e9;
  printf("tick %3.0f ns  residues 61/64/67%s: from one pair %d wrong; with the vote %d wrong of %d%s   dot rel err %.1e   %.0f weights/s on one line\n",
    tick, tri?" + tri-edge":"", raw_wrong, wrong, N, tri?({static char b[64]; snprintf(b,64,"   lap flags wrong %d",lapwrong); b;}):"", fabs(acc-ref)/fabs(ref), N/secs);
  return 0; }
