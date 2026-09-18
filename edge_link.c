// STEP 1 ON A REAL LINK. Two processes. The sender holds weights as ticks and
// fires one edge per weight, at its tick, by writing a shared flag. The
// receiver never sees a value: it timestamps each edge against the shared
// clock, reads the dial, and accumulates dial x input. Then the nested radix:
// a weight as two edges on a 64-dial (12 bits in 128 slots instead of 4096).
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
static void realtime(void){   // ask the scheduler not to pause us: time-constraint policy
  thread_time_constraint_policy_data_t pol={.period=0,.computation=50000,.constraint=100000,.preemptible=0};
  thread_policy_set(pthread_mach_thread_np(pthread_self()),THREAD_TIME_CONSTRAINT_POLICY,(thread_policy_t)&pol,THREAD_TIME_CONSTRAINT_POLICY_COUNT); }
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define N 768
int main(int argc,char**argv){
  double tick=argc>1?atof(argv[1]):1000;        // ns per tick
  int radix=argc>2?atoi(argv[2]):4096;          // 4096 = flat; 64 = two nested 64-dials
  int twin=argc>3?atoi(argv[3]):0;              // 1 = send every edge twice, 180 degrees apart
  int digits = (radix==4096?1:2)*(twin?2:1);
  volatile long*flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile double*tstamp=mmap(0,N*digits*8+64,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile double*pilot=tstamp+N*digits+2;
  srand(7); float w[N],x[N]; float cyc=0; for(int i=0;i<N;i++){w[i]=((float)rand()/RAND_MAX-0.5f)*0.2f; x[i]=(float)rand()/RAND_MAX-0.5f; if(fabsf(w[i])>cyc)cyc=fabsf(w[i]);}
  int half=2047; int a[N]; for(int i=0;i<N;i++){int q=(int)lroundf(w[i]/cyc*half); a[i]=((q%4096)+4096)%4096;}
  double turn = tick*(digits==1?4096:64);
  pid_t pid=fork();
  if(pid==0){ // RECEIVER: sees edges only
    if(getenv("RT")) realtime();
    long seen=0; int k=0; while(*flag==seen); seen=*flag; *pilot=ns();          // the pilot edge
    while(k<N*digits){ while(*flag==seen); seen=*flag; tstamp[k++]=ns(); } _exit(0); }
  if(getenv("RT")) realtime();
  usleep(20000); *flag=0; double t0=ns()+200000; tstamp[N*digits]=t0; long e=0;
  while(ns()<t0); *flag=++e;                                                      // pilot at exactly t0
  t0 += 100000;                                                                   // weights start 100 us later
  // SENDER: weight i occupies turn i; fire at its tick within that turn (two turns if two digits)
  for(int i=0;i<N;i++) for(int d=0;d<digits;d++){
    int dd = twin ? d/2 : d, m = radix==4096 ? 4096 : 64;
    int digit = radix==4096 ? a[i] : (dd==0 ? a[i]/64 : a[i]%64);
    if(twin && (d&1)) digit = (digit + m/2) % m;                       // the twin, half a dial later
    double at = t0 + (double)(i*digits+d)*turn + digit*tick;
    while(ns()<at); *flag=++e; }
  int st; wait(&st);
  // RECEIVER-SIDE DECODE (done here after the fact from its timestamps): read the dial at arrival
  double delay=*pilot-(t0-100000);                                                // measured trip, subtracted below
  for(int k=0;k<N*digits;k++) tstamp[k]-=delay;
  double acc=0, ref=0; int bad=0, maxoff=0, disagree=0;
  int nd = radix==4096?1:2, m = radix==4096?4096:64;
  for(int i=0;i<N;i++){ int val=0;
    for(int dd=0;dd<nd;dd++){ int r;
      if(!twin){ int d=dd; double t=tstamp[i*digits+d]-t0-(double)(i*digits+d)*turn; r=(int)floor(t/tick+0.5); }
      else { int d=2*dd; double t1=tstamp[i*digits+d]-t0-(double)(i*digits+d)*turn, t2=tstamp[i*digits+d+1]-t0-(double)(i*digits+d+1)*turn;
             double r1=t1/tick, r2=t2/tick-m/2;                       // rotate the twin back 180 degrees
             double diff=r1-r2; diff-=m*floor(diff/m+0.5);            // shortest way round
             if(fabs(diff)>2) disagree++;                              // the twins do not agree: a slip
             r=(int)floor((r1-diff/2)+0.5); }                          // average of the two readings
      r=((r%m)+m)%m; val=val*(radix==4096?1:64)+r; }
    int off=abs(val-a[i]); if(off>2048) off=4096-off; if(off) bad++; if(off>maxoff)maxoff=off;
    int s=((val+2048)%4096)-2048; acc+=(double)s/half*x[i]; ref+=(double)w[i]/cyc*x[i]; }
  if(twin) printf("  [twins disagreed on %d weights] ", disagree);
  printf("trip %.0f ns | tick %5.0f ns  %s%s: %d edges for %d weights, turn %.1f us, %d weights wrong (worst off by %d ticks), dot product %.5f vs %.5f (rel err %.1e), %.0f weights/s on one line\n",
    delay, tick, radix==4096?"flat 4096":"nested 64x64", twin?" twin":"", N*digits, N, turn/1e3, bad, maxoff, acc, ref, fabs(acc-ref)/fabs(ref), 1e9/(turn*digits));
  return 0; }
