// THE CHAIN. Two sides, one dial, layers alternating sides.
//   side A holds the notes of layers 1 and 3, side B the notes of layers 2 and 4.
//   round r: the activations cross (few edges, one per value, laps as magnitude)
//            to the side that holds the next notes' PARTNER -- no: the notes
//            cross to the side that holds the activations. So:
//     L1: A fires its notes -> B adds into x        (x lives at B)   -> y1 at B, GELU
//     L2: B holds L2 notes; y1 is at B too. B fires y1 across to A as edges;
//         then B fires L2's notes to A; A adds -> y2 at A, GELU
//     L3: A holds L3 notes; y2 at A. A fires y2 to B, then L3 notes; B adds -> y3
//     L4: B fires y3 to A, then L4 notes; A adds -> y4 = the answer, at A.
//   Every crossing is edges only. Notes are sent 3x, median-voted, on a sorted
//   wiring both sides know. Activations are sent 3x too.
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
#define D 64
#define L 4
#define HALF 2047
#define PILOTS 10
#define COPIES 3
static double tick=400; static int G=4; static int SECTIONS=1;

// ---------------------------------------------------------------- lines
typedef struct { volatile long*flag; volatile double*ts; volatile long*dl; long cap; } line_t;
static line_t mk(long cap){ line_t l; l.flag=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0); l.ts=mmap(0,(cap+64)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0); l.dl=mmap(0,(cap+64)*8,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0); l.cap=cap; return l; }
// receive n edges (plus pilots): stamp arrivals, record counter deltas
static long recv(line_t*l, long n, long*seen_io){ long seen=*seen_io, k=0; double tw=ns();
  while(k<n){ long f; while((f=*l->flag)==seen){ if(ns()-tw>4e9) goto out; } l->dl[k]=f-seen; seen=f; l->ts[k++]=ns(); if(k%4096==0) tw=ns(); }
  out: *seen_io=seen; return k; }
// send a sequence of (gap ticks, counter step) as edges; pilots first
static void send(line_t*l, const long*gaps, const long*steps, long n, long*ctr){ double t=ns()+100000; long e=*ctr;
  for(int p=0;p<PILOTS;p++){ while(ns()<t+p*100*tick); *l->flag=++e; }
  double at=t+(PILOTS-1)*100*tick;
  for(long i=0;i<n;i++){ at+=(gaps[i]+1+G)*tick; while(ns()<at); e+=steps[i]; *l->flag=e; }
  usleep(500); *l->flag=++e; *ctr=e; }                                   // trailing edge closes the stream

// ---------------------------------------------------------------- decode helpers (receiver side)
// align stamps by counter; returns arrays tsa/have of length n_expected+PILOTS+1
static void align(line_t*l, long got, long expect, double*tsa, char*have, long*sym){ memset(have,0,expect); long c=0;
  for(long m=0;m<got;m++){ c+=labs(l->dl[m])>0? (l->dl[m]>0?1:-1)*1:0; /* counter direction only for position */ }
  // simpler: positions are cumulative |delta| == 1 for pilots/activations/notes since steps are +-1 (notes) or +-(1+laps) (activations)
  c=0; for(long m=0;m<got;m++){ long d=l->dl[m]; long adv = (labs(d)>0)?1:0; (void)adv; c+= (d>0? d : -d) ; }
  // we encode every stream so that |counter step| == 1 for position; magnitude info goes in a parallel word: see below
}

// The counter carries POSITION (+1 per edge); a second shared word carries the symbol (sign & laps) written just before the edge.
// (On real hardware that second word is the edge's direction/magnitude; here it is a neighbour cache line.)
static long notes_decode(line_t*l, long got, long n, const long*order_steps_unused, long*val_out, char*ok_out, int copies){
  // stamps -> sorted-step values per copy; median across copies
  static double tsa[3*D*D+PILOTS+8]; static char have[3*D*D+PILOTS+8]; memset(have,0,sizeof have);
  long c=0; for(long m=0;m<got;m++){ c+=l->dl[m]; if(c>=1&&c<=n*copies+PILOTS+1){ tsa[c-1]=l->ts[m]; have[c-1]=1; } }
  long missed=0; for(long m=0;m<n*copies+PILOTS;m++) if(!have[m]) missed++;
  static long note[3][D*D]; static char ok[3][D*D];
  for(int p=0;p<copies;p++){ long idx=PILOTS+(long)p*n; long val=-HALF; long prev=idx-1; while(!have[prev]&&prev>0) prev--;
    for(long i=0;i<n;i++){ ok[p][i]=0; if(i%D==0) val=-HALF;   /* every neuron's steps start from the far side of the dial */
      if(have[idx]){ long span=idx-prev; long st=(long)floor((tsa[idx]-tsa[prev])/tick+0.5)-span*(1+G); val+=st; prev=idx; if(span==1){ note[p][i]=val; ok[p][i]=1; } } idx++; } }
  for(long i=0;i<n;i++){ long v[3]; int k=0; for(int p=0;p<copies;p++) if(ok[p][i]) v[k++]=note[p][i];
    if(k==0){ ok_out[i]=0; continue; } ok_out[i]=1;
    if(k==1) val_out[i]=v[0]; else if(k==2) val_out[i]=(v[0]==v[1])?v[0]:v[0]; else { long a=v[0],b=v[1],c2=v[2]; val_out[i]=(a<=b)?((b<=c2)?b:((a<=c2)?c2:a)):((a<=c2)?a:((b<=c2)?c2:b)); } }
  return missed; }

int main(int argc,char**argv){
  if(argc>1) tick=atof(argv[1]); if(argc>2) G=atoi(argv[2]); if(argc>3) SECTIONS=atoi(argv[3]);
  // ---- four real 64x64 slices: h5 c_fc, h5 c_proj, h6 c_fc, h6 c_proj (inputs 0..63, outputs 0..63), 12-bit angles
  const char*files[L]={"h5_fc_w.i16","h5_mproj_w.i16","h6_fc_w.i16","h6_mproj_w.i16"}; int nin_full[L]={768,3072,768,3072};
  static int q[L][D][D];
  for(int l=0;l<L;l++){ char path[256]; snprintf(path,256,"/Users/abundancemachine/tick-machine/gpt2_dial/%s",files[l]);
    FILE*f=fopen(path,"rb"); int16_t*w=malloc((size_t)4096*3072*2); size_t nr=fread(w,2,(size_t)4096*3072,f); fclose(f); (void)nr;
    for(int j=0;j<D;j++) for(int i=0;i<D;i++) q[l][j][i]=(int)lround(w[(size_t)j*nin_full[l]+i]/16.0); free(w); }
  // sorted wiring per neuron per layer (both sides know it); gaps and steps for the notes
  static int order[L][D][D]; static long gaps[L][D*D];
  for(int l=0;l<L;l++) for(int j=0;j<D;j++){ for(int i=0;i<D;i++) order[l][j][i]=i;
    for(int i=1;i<D;i++){ int k=order[l][j][i],m=i-1; while(m>=0&&q[l][j][order[l][j][m]]>q[l][j][k]){order[l][j][m+1]=order[l][j][m];m--;} order[l][j][m+1]=k; }
    gaps[l][j*D+0]=q[l][j][order[l][j][0]]+HALF; for(int i=1;i<D;i++) gaps[l][j*D+i]=q[l][j][order[l][j][i]]-q[l][j][order[l][j][i-1]]; }
  static long ones[3*D*D]; for(int i=0;i<3*D*D;i++) ones[i]=1;
  // input on the dial
  srand(5); static long x0[D]; for(int i=0;i<D;i++){ double u=rand()/(double)RAND_MAX,v=rand()/(double)RAND_MAX; x0[i]=lround(sqrt(-2*log(u+1e-12))*cos(6.2831853*v)*600); }
  // ---- the reference chain, integer, same GELU and requantisation the sides use
  static long ref[L+1][D]; memcpy(ref[0],x0,sizeof x0);
  for(int l=0;l<L;l++){ double mx=0; static double y[D];
    for(int j=0;j<D;j++){ long s=0; for(int i=0;i<D;i++) s+=(long)q[l][j][i]*ref[l][i]; double v=s/2047.0; y[j]=0.5*v*(1+tanh(0.7978845608*(v+0.044715*v*v*v))); if(fabs(y[j])>mx) mx=fabs(y[j]); }
    for(int j=0;j<D;j++) ref[l+1][j]=lround(y[j]/mx*HALF); }
  // ---- lines
  line_t AB=mk(3*D*D+PILOTS+8), BA=mk(3*D*D+PILOTS+8);
  volatile long*report=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);   // B's per-layer results back to the parent
  pid_t pid=fork();
  int me = pid==0 ? 1 : 0;                                   // 0 = side A (parent), 1 = side B (child)
  line_t*in = me? &AB : &BA; line_t*out = me? &BA : &AB;
  long ctr_out=0, seen_in=0; static long act[D]; static long got_notes[D*D]; static char okn[D*D];
  if(me==1){ alarm(60); memcpy(act,x0,sizeof act); }        // B starts with the input
  double t_start=ns();
  if(SECTIONS){
    // who runs which layer, locally: A: layer 1; B: layers 2,3; A: layer 4.   1 -> (2,3) -> back (4)
    int owner[L]={0,1,1,0}; static long cur[D]; if(me==0) memcpy(cur,x0,sizeof cur);
    long hops=0;
    for(int l=0;l<L;l++){
      if(l>0 && owner[l]!=owner[l-1]){                      // the activations cross, as edges, 3 copies
        if(me==owner[l-1]){ static long ag[3*D], as[3*D]; for(int c=0;c<COPIES;c++) for(int i=0;i<D;i++){ long v=cur[i]; long m=labs(v); ag[c*D+i]=m%4096; as[c*D+i]=(v<0?-1:1)*(1+m/4096); } send(out,ag,as,3*D,&ctr_out); }
        else { long got=recv(in,3*D+PILOTS+1,&seen_in); static double ta[3*D+PILOTS+8]; static char ha[3*D+PILOTS+8]; static long sa[3*D+PILOTS+8]; memset(ha,0,sizeof ha);
          long c=0; for(long m=0;m<got;m++){ c+=1; if(c<=3*D+PILOTS+1){ ta[c-1]=in->ts[m]; sa[c-1]=in->dl[m]; ha[c-1]=1; } }
          for(int i=0;i<D;i++){ long v[3]; int k=0; for(int cp=0;cp<COPIES;cp++){ long idx=PILOTS+cp*D+i; if(ha[idx]&&ha[idx-1]){ long angle=(long)floor((ta[idx]-ta[idx-1])/tick+0.5)-1-G; long sgn=sa[idx]<0?-1:1; long laps=labs(sa[idx])-1; v[k++]=sgn*(angle+4096*laps); } }
            cur[i]= k==0?0 : (k<3? v[0] : ((v[0]<=v[1])?((v[1]<=v[2])?v[1]:((v[0]<=v[2])?v[2]:v[0])):((v[0]<=v[2])?v[0]:((v[1]<=v[2])?v[2]:v[1])))); }
          int ex=0; for(int i=0;i<D;i++) if(cur[i]==ref[l][i]) ex++; report[l*4+2]=ex; }
        hops++; }
      if(me==owner[l]){                                       // this side runs the layer from its OWN notes: nothing crosses
        static double y[D]; double mx=0;
        for(int j=0;j<D;j++){ long sum=0; for(int i=0;i<D;i++) sum+=(long)q[l][j][i]*cur[i]; double vv=sum/2047.0; y[j]=0.5*vv*(1+tanh(0.7978845608*(vv+0.044715*vv*vv*vv))); if(fabs(y[j])>mx) mx=fabs(y[j]); }
        for(int j=0;j<D;j++) cur[j]=lround(y[j]/mx*HALF);
        int ex=0; for(int j=0;j<D;j++) if(cur[j]==ref[l+1][j]) ex++; report[l*4+0]=ex; report[l*4+1]=owner[l]; }
    }
    if(me==1) _exit(0);
    int st; wait(&st); double secs=(ns()-t_start)/1e9;
    printf("THE CHAIN (sections)  tick %.0f ns: A runs layer 1 -> B runs 2,3 -> A runs 4.  notes never move; only activations cross, 3 copies as edges\n", tick);
    for(int l=0;l<L;l++){ printf("  layer %d at side %s: output exact vs reference %ld/64", l+1, report[l*4+1]?"B":"A", report[l*4+0]); if(l>0 && ((l==1)||(l==3))) printf("   (activations arrived exact %ld/64)", report[l*4+2]); printf("\n"); }
    printf("  [MEASURED] final answer at A: %ld of 64 identical to the reference; %ld hops, %d activation edges per hop, %.2f s\n", report[(L-1)*4+0], hops, 3*D, secs);
    return 0; }
  for(int l=0;l<L;l++){
    int holder = l%2;                                        // who holds this layer's notes: even layers A, odd layers B
    int adder = 1-holder;                                    // the other side adds (it must have the activations)
    if(me==holder){
      // if the activations are on MY side, send them to the adder first (3 copies, laps as counter step)
      if(l>0){ static long ag[3*D], as[3*D]; for(int c=0;c<COPIES;c++) for(int i=0;i<D;i++){ long v=act[i]; long m=labs(v); ag[c*D+i]=m%4096; as[c*D+i]=(v<0?-1:1)*(1+m/4096); }
               send(out,ag,as,3*D,&ctr_out); }
      // then the notes, 3 copies
      static long g3[3*D*D]; for(int c=0;c<COPIES;c++) memcpy(g3+c*D*D,gaps[l],sizeof(long)*D*D);
      send(out,g3,ones,3*D*D,&ctr_out);
    } else {
      long lapsum=0;
      if(l>0){ // receive activations: 3 copies, counter step carries sign & laps, gap carries angle
        long got=recv(in,3*D+PILOTS+1,&seen_in); static double ta[3*D+PILOTS+8]; static char ha[3*D+PILOTS+8]; static long sa[3*D+PILOTS+8]; memset(ha,0,sizeof ha);
        long c=0; for(long m=0;m<got;m++){ long d=in->dl[m]; c+=1; if(c>=1&&c<=3*D+PILOTS+1){ ta[c-1]=in->ts[m]; sa[c-1]=d; ha[c-1]=1; } }
        for(int i=0;i<D;i++){ long v[3]; int k=0; for(int cp=0;cp<COPIES;cp++){ long idx=PILOTS+cp*D+i; if(ha[idx]&&ha[idx-1]){ long angle=(long)floor((ta[idx]-ta[idx-1])/tick+0.5)-1-G; long sgn=sa[idx]<0?-1:1; long laps=labs(sa[idx])-1; v[k++]=sgn*(angle+4096*laps); } }
          act[i]= k==0?0 : (k==1? v[0] : (k==2? v[0] : ((v[0]<=v[1])?((v[1]<=v[2])?v[1]:((v[0]<=v[2])?v[2]:v[0])):((v[0]<=v[2])?v[0]:((v[1]<=v[2])?v[2]:v[1]))))); }
        (void)lapsum; }
      // receive the notes and ADD
      long got=recv(in,3*D*D+PILOTS+1,&seen_in);
      long missed=notes_decode(in,got,D*D,NULL,got_notes,okn,COPIES);
      long nwrong=0; for(int j=0;j<D;j++) for(int i=0;i<D;i++){ long want=q[l][j][order[l][j][i]]; if(!okn[j*D+i] || got_notes[j*D+i]!=want) nwrong++; }
      if(l==0){ fprintf(stderr,"  [B round 1] first notes decoded vs sent: "); for(int i=0;i<6;i++) fprintf(stderr,"%ld/%d ",got_notes[i],q[0][0][order[0][0][i]]); fprintf(stderr," ... notes wrong %ld of %d\n",nwrong,D*D); }
      report[l*4+3]=nwrong;
      static double y[D]; double mx=0; int wrong=0;
      for(int j=0;j<D;j++){ long s=0; for(int i=0;i<D;i++){ long v = okn[j*D+i]? got_notes[j*D+i] : 0; s+=v*act[order[l][j][i]]; }
        long sref=0; for(int i=0;i<D;i++) sref+=(long)q[l][j][i]*act[i]; if(s!=sref) wrong++;
        double vv=s/2047.0; y[j]=0.5*vv*(1+tanh(0.7978845608*(vv+0.044715*vv*vv*vv))); if(fabs(y[j])>mx) mx=fabs(y[j]); }
      for(int j=0;j<D;j++) act[j]=lround(y[j]/mx*HALF);                    // the new activations, on the dial, here
      int exact=0; for(int j=0;j<D;j++) if(act[j]==ref[l+1][j]) exact++;
      report[l*4+0]=exact; report[l*4+1]=wrong; report[l*4+2]=missed;
    }
  }
  if(me==1) _exit(0);
  int st; wait(&st); double secs=(ns()-t_start)/1e9;
  printf("THE CHAIN  tick %.0f ns, guard %d, 4 real GPT-2 64x64 slices, notes on alternating sides, %d copies + median vote\n", tick, G, COPIES);
  for(int l=0;l<L;l++) printf("  round %d: layer %d notes fired %s -> %s adds   notes wrong %ld of %d (missed %ld)   sums exact %ld/64   activations exact vs reference %ld/64\n",
     l+1, l+1, l%2==0?"A":"B", l%2==0?"B":"A", report[l*4+3], D*D, report[l*4+2], 64-report[l*4+1], report[l*4+0]);
  printf("  [MEASURED] final answer at side %s: %ld of 64 values identical to the integer reference chain; %.2f s total, %ld edges crossed (notes 3x + activations 3x), nothing else\n",
     L%2==0?"A":"B", report[(L-1)*4+0], secs, (long)L*3*D*D + (long)(L-1)*3*D);
  return 0; }
