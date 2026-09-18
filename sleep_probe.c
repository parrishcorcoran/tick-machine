#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
int main(){
  long asks[]={0,1,100,1000,10000,100000,1000000};
  printf("what a sleep actually lasts (2000 tries each):\n");
  for(int k=0;k<7;k++){ struct timespec ts={0,asks[k]}; double mn=1e18,mx=0,sum=0;
    for(int i=0;i<2000;i++){ double t0=ns(); nanosleep(&ts,0); double d=ns()-t0; sum+=d; if(d<mn)mn=d; if(d>mx)mx=d; }
    printf("  ask %8ld ns -> min %9.0f  mean %9.0f  max %9.0f ns\n", asks[k], mn, sum/2000, mx); }
  // mach_wait_until: the OS's absolute-time wait
  printf("mach_wait_until to an instant 10 us ahead: ");
  double mn=1e18,mx=0,sum=0; for(int i=0;i<2000;i++){ uint64_t t=mach_absolute_time()+10000*3/125; double t0=ns(); mach_wait_until(t); double d=ns()-t0-10000; sum+=d; if(d<mn)mn=d; if(d>mx)mx=d; }
  printf("late by min %.0f mean %.0f max %.0f ns\n", mn, sum/2000, mx);
  return 0; }
