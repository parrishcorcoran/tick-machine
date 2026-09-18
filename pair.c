// THE PAIR. Two sides, one dial, bits both ways.
//   side A (storage): holds notes. forward cycler fires each note as one edge at
//                     its angle (sorted steps, order = wiring). reverse cycler
//                     receives the answers.
//   side B (compute): reverse cycler stamps each incoming edge, reads the dial,
//                     adds into the neuron's ADDER. when a neuron's notes are
//                     all in, its forward cycler sends the SUM back as one edge:
//                     angle = sum mod 4096 in timing, laps = edge magnitude.
// Real GPT-2 notes (from gpt2_dial export), real input. A decodes the returned
// sums and checks them against the matmul. Nothing but edges cross.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define NIN 768
#define NEUR 64
#define HALF 2047
static int G=4;                  // guard ticks after every edge (argv[2])
#define PILOTS 10
int main(int argc,char**argv){
  double tick=argc>1?atof(argv[1]):200; if(argc>2) G=atoi(argv[2]);
  // ---- real notes: block 5 mlp.c_fc, first NEUR neurons, 12-bit angles (the 16-bit export / 16)
  int16_t*w16=malloc(3072*NIN*2); FILE*f=fopen("/Users/abundancemachine/tick-machine/gpt2_dial/h5_fc_w.i16","rb"); fread(w16,2,3072*NIN,f); fclose(f);
  float cyc[3072]; f=fopen("/Users/abundancemachine/tick-machine/gpt2_dial/h5_fc_cyc.f32","rb"); fread(cyc,4,3072,f); fclose(f);
  static int q[NEUR][NIN]; for(int j=0;j<NEUR;j++) for(int i=0;i<NIN;i++) q[j][i]=(int)lround(w16[j*NIN+i]/16.0);
  // real-ish input on the dial
  srand(11); static int xq[NIN]; for(int i=0;i<NIN;i++){ double u=rand()/(double)RAND_MAX,v=rand()/(double)RAND_MAX; xq[i]=(int)lround(sqrt(-2*log(u+1e-12))*cos(6.2831853*v)*600); }
  // the wiring: each neuron's notes sorted once; both sides know the order
  static int order[NEUR][NIN], step[NEUR][NIN];
  for(int j=0;j<NEUR;j++){ for(int i=0;i<NIN;i++) order[j][i]=i;
    for(int i=1;i<NIN;i++){ int k=order[j][i],m=i-1; while(m>=0&&q[j][order[j][m]]>q[j][k]){order[j][m+1]=order[j][m];m--;} order[j][m+1]=k; }
    step[j][0]=q[j][order[j][0]]+HALF; for(int i=1;i<NIN;i++) step[j][i]=q[j][order[j][i]]-q[j][order[j][i-1]]; }
  // ---- the two lines (shared flags), timestamps
  volatile long*lineAB=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);   // A -> B
  volatile long*lineBA=lineAB+16;                                                     // B -> A (own cache line)
  volatile double*tsB=mmap(0,(3*NEUR*NIN+PILOTS+8)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*dB=mmap(0,(3*NEUR*NIN+PILOTS+8)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile double*tsA=mmap(0,(NEUR+PILOTS+8)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*dA=mmap(0,(NEUR+PILOTS+8)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
  volatile long*ready=lineAB+32;
  long total_in=3*NEUR*NIN+PILOTS+1;   // three passes of the notes
  pid_t pid=fork();
  if(pid==0){ // ===== SIDE B: compute
    alarm(8);
    long seen=0; long k=0;
    double tw=ns(); while(seen<total_in){ long fl; while((fl=*lineAB)==seen){ if(ns()-tw>3e9) goto done_rx; } dB[k]=fl-seen; seen=fl; tsB[k++]=ns(); }   // reverse cycler: stamp every edge; stop by COUNT, so a missed edge cannot hang it
    done_rx:;
    // ADDER: decode each note from timing (absolute from the last pilot), add into its neuron
    static double tsa[3*NEUR*NIN+PILOTS+8]; static char have[3*NEUR*NIN+PILOTS+8]; long c=0; long missed=0;
    for(long m=0;m<k;m++){ c+=dB[m]; if(c>=1&&c<=total_in){ tsa[c-1]=tsB[m]; have[c-1]=1; } }
    for(long m=0;m<total_in-1;m++) if(!have[m]) missed++;
    static long note[3][NEUR][NIN]; static char ok[3][NEUR][NIN];
    for(int pass=0;pass<3;pass++){ long idx=PILOTS+(long)pass*NEUR*NIN;
      for(int j=0;j<NEUR;j++){ long val=-HALF; long prev=idx-1; while(!have[prev]&&prev>0) prev--;
        for(int i=0;i<NIN;i++){
          if(have[idx]){ long span=idx-prev; long st=(long)floor((tsa[idx]-tsa[prev])/tick+0.5)-span*(1+G); val+=st; prev=idx;
            if(span==1){ note[pass][j][i]=val; ok[pass][j][i]=1; } }
          idx++; } } }
    static long acc[NEUR]; long unfilled=0;
    long disagreed=0;
    for(int j=0;j<NEUR;j++){ long s=0; for(int i=0;i<NIN;i++){
        long v[3]; int n=0; for(int p=0;p<3;p++) if(ok[p][j][i]) v[n++]=note[p][j][i];
        long pick; if(n==0){ unfilled++; continue; }
        if(n>=2 && v[0]==v[1]) pick=v[0]; else if(n==3 && v[1]==v[2]) pick=v[1]; else if(n==3 && v[0]==v[2]) pick=v[0]; else { if(n==3){ long a=v[0],b=v[1],c=v[2]; pick = (a<=b)?((b<=c)?b:((a<=c)?c:a)):((a<=c)?a:((b<=c)?c:b)); } else pick=v[0]; if(n>=2) disagreed++; }   /* median: a one-tick slip is symmetric */
        s+=pick*xq[order[j][i]]; } acc[j]=s; }
    *(lineAB+64)=disagreed;
    *(lineAB+56)=unfilled;
    *(lineAB+48)=missed;
    // forward cycler: send each sum back as ONE edge: angle = sum mod 4096 (timing), laps = magnitude of the step
    *ready=1; usleep(20000); double t=ns()+200000; long e=0;
    for(int p=0;p<PILOTS;p++){ while(ns()<t+p*100*tick); *lineBA=++e; }
    double at=t+(PILOTS-1)*100*tick;
    for(int j=0;j<NEUR;j++){ long v=acc[j]; long sign=v<0?-1:1; long m=labs(v); long angle=m%4096, laps=m/4096;
      at+=(angle+1+G)*tick; while(ns()<at); e+=sign*(1+laps); *lineBA=e; }
    _exit(0); }
  // ===== SIDE A: storage
  usleep(20000); double t0=ns()+200000; long e=0;
  for(int p=0;p<PILOTS;p++){ while(ns()<t0+p*100*tick); *lineAB=++e; }
  double at=t0+(PILOTS-1)*100*tick; double tsend=ns();
  for(int pass=0;pass<3;pass++) for(int j=0;j<NEUR;j++) for(int i=0;i<NIN;i++){ at+=(step[j][i]+1+G)*tick; while(ns()<at); *lineAB=++e; }   // forward cycler, twice
  double tdone=ns(); usleep(1000); *lineAB=++e;
  // reverse cycler: receive the sums
  double tr=ns(); while(!*ready){ if(ns()-tr>3e9){ printf("  side B never answered\n"); break; } } long seen=0; int k=0; double tA0=ns();
  while(k<NEUR+PILOTS){ long fl; while((fl=*lineBA)==seen){ if(ns()-tA0>3e9) break; } if(fl==seen) break; dA[k]=fl-seen; seen=fl; tsA[k++]=ns(); }
  int st; wait(&st);
  // decode: sum_j = sign * (angle + 4096*laps)
  double base=tsA[PILOTS-1]; int exact=0; long worst=0; double rel=0, refn=0;
  for(int j=0;j<NEUR;j++){ int idx=PILOTS+j; if(idx>=k){ printf("  missing answer for neuron %d\n",j); break; }
    double gap=(tsA[idx]-tsA[idx-1])/tick; long angle=(long)floor(gap+0.5)-1-G; long sign=dA[idx]<0?-1:1; long laps=labs(dA[idx])-1;
    long got=sign*(angle+4096*laps);
    long ref=0; for(int i=0;i<NIN;i++) ref+=(long)q[j][i]*xq[i];
    if(got==ref) exact++; else if(labs(got-ref)>worst) worst=labs(got-ref);
    rel+=(double)(got-ref)*(got-ref); refn+=(double)ref*ref; }
  printf("tick %4.0f ns  |  A -> B: %d notes x3 as %d edges in %.1f ms (%.0f edges/s), B missed %ld, unfilled %ld, unresolved votes %ld  |  B -> A: %d sums as %d edges, A got %d\n",
    tick, NEUR*NIN, 3*NEUR*NIN, (tdone-tsend)/1e6, 3*NEUR*NIN/((tdone-tsend)/1e9), *(lineAB+48), *(lineAB+56), *(lineAB+64), NEUR, NEUR, k-PILOTS);
  printf("  [MEASURED] sums exact: %d of %d   worst off by %ld   rel err %.1e   (real GPT-2 h5.mlp.c_fc notes, 12-bit dial, laps as edge magnitude)\n",
    exact, NEUR, worst, sqrt(rel/refn));
  return 0; }
