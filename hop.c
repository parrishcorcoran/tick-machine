// One edge, core to core, measured one way against the shared clock.
// Sender writes its send-time and flips a flag; receiver spins, reads the
// clock on the flip, and records (arrival - send). The DISTRIBUTION of that
// number is the usable tick: a constant delay subtracts away; the spread does not.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){
  int n=argc>1?atoi(argv[1]):20000; double gap=argc>2?atof(argv[2]):2000;   // ns between edges
  volatile long*flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile double*sent=(volatile double*)(flag+8);
  double*trip=mmap(0,n*8+4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  *flag=0; pid_t pid=fork();
  if(pid==0){ long seen=0; int k=0; while(seen<n){ while(*flag==seen); seen=*flag; double t=ns(); trip[k++]=t-*sent; } for(;k<n;k++)trip[k]=trip[k-1]; _exit(0); }
  usleep(20000); double next=ns()+100000;
  for(long i=1;i<=n;i++){ while(ns()<next); *sent=ns(); *flag=i; next+=gap; }
  int st; wait(&st);
  qsort(trip,n,8,cmp);
  double med=trip[n/2];
  printf("%d edges, one per %.1f us, core to core:\n", n, gap/1e3);
  printf("  trip: min %.0f ns  10%% %.0f  median %.0f  90%% %.0f  99%% %.0f  99.9%% %.0f  max %.0f ns\n",
    trip[0],trip[n/10],med,trip[n*9/10],trip[n*99/100],trip[n*999/1000],trip[n-1]);
  int w42=0,w100=0,w200=0; for(int i=0;i<n;i++){ double d=trip[i]-med; if(d<0)d=-d; w42+=d<=21; w100+=d<=50; w200+=d<=100; }
  printf("  edges landing within half a tick of the median:  42 ns tick %.1f%%   100 ns tick %.1f%%   200 ns tick %.1f%%\n",
    100.0*w42/n,100.0*w100/n,100.0*w200/n);
  return 0; }
