// Edge jitter across the two links a CPU offers: a pipe, and a shared-memory flag.
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
int main(){
  // pipe
  int p[2]; pipe(p); pid_t pid=fork();
  if(pid==0){ close(p[1]); unsigned char b; double prev=0,sumd=0,mind=1e18,maxd=0; int m=0;
    while(read(p[0],&b,1)==1){ double t=ns(); if(prev){double d=t-prev;sumd+=d;if(d<mind)mind=d;if(d>maxd)maxd=d;m++;} prev=t; }
    printf("  pipe   : edges sent every 50.0 us arrive spaced mean %.2f us, min %.2f, max %.2f  (jitter %.1f us)\n",sumd/m/1e3,mind/1e3,maxd/1e3,(maxd-mind)/1e3); fflush(stdout); _exit(0); }
  close(p[0]); unsigned char b=1; double next=ns()+100000;
  for(int i=0;i<2000;i++){ while(ns()<next); write(p[1],&b,1); next+=50000; }
  close(p[1]); int st; wait(&st);
  // shared-memory flag: sender flips a counter at scheduled instants; receiver spins
  volatile long *flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0); *flag=0;
  for(int period_ns=50000; period_ns>=500; period_ns/=10){
    pid=fork();
    if(pid==0){ long seen=0; double prev=0,sumd=0,mind=1e18,maxd=0; int m=0;
      while(seen<2000){ while(*flag==seen); seen=*flag; double t=ns(); if(prev){double d=t-prev;sumd+=d;if(d<mind)mind=d;if(d>maxd)maxd=d;m++;} prev=t; }
      printf("  shm    : edges every %6.1f us arrive spaced mean %8.3f us, min %8.3f, max %8.3f  (jitter %.2f us)\n",period_ns/1e3,sumd/m/1e3,mind/1e3,maxd/1e3,(maxd-mind)/1e3); fflush(stdout); _exit(0); }
    *flag=0; usleep(20000); double nxt=ns()+100000;
    for(long i=1;i<=2000;i++){ while(ns()<nxt); *flag=i; nxt+=period_ns; }
    wait(&st);
  }
  return 0; }
