// The line, as a library both sections call. A file-backed shared page holds the
// counter (the flag). One value = one edge: timing carries value mod 64, the
// counter step carries sign and laps (+-(1 + |v|/64)). Copies + median vote.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <sys/mman.h>
#include <mach/mach_time.h>
static inline double ns(){return (double)mach_absolute_time()*125.0/3.0;}
#define PILOTS 10
#define G 4
#define DIAL 64
static volatile long* line(const char*path){ int fd=open(path,O_RDWR|O_CREAT,0666); ftruncate(fd,4096); volatile long*p=mmap(0,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd); return p; }

int edge_recv2(const char*path, long*out, int n, double tick, int copies, double timeout_s, long*missing, int maxmiss);
int edge_send(const char*path, const long*vals, int n, double tick, int copies){
  volatile long*flag=line(path); volatile long*ready=flag+16;
  double tw=ns(); while(!*ready){ if(ns()-tw>60e9) return -1; }   /* wait until the receiver is listening */
  *ready=0; long e=*flag; volatile long*seqw=flag+32; long drop=getenv("EDGE_DROP")?atol(getenv("EDGE_DROP")):-1;
  double t=ns()+100000;
  for(int p=0;p<PILOTS;p++){ while(ns()<t+p*100*tick); *seqw=-2; *flag=++e; }
  double at=t+(PILOTS-1)*100*tick;
  for(int c=0;c<copies;c++) for(int i=0;i<n;i++){ long v=vals[i]; long m=labs(v); long angle=m%DIAL; long step=(v<0?-1:1)*(1+m/DIAL);
    at+=(angle+1+G)*tick; while(ns()<at); if(drop>=0 && c*n+i==drop){ e+=step; continue; } *seqw=c*n+i; e+=step; *flag=e; }   /* EDGE_DROP=k: test hook, swallow edge k */
  usleep(500); *seqw=-1; *flag=++e;                         // trailing edge
  return copies*n+PILOTS+1; }

// returns number of values decoded; out[i] = median over copies
int edge_recv(const char*path, long*out, int n, double tick, int copies, double timeout_s){ return edge_recv2(path,out,n,tick,copies,timeout_s,NULL,0); }
int edge_recv2(const char*path, long*out, int n, double tick, int copies, double timeout_s, long*missing, int maxmiss){
  volatile long*flag=line(path); volatile long*ready=flag+16; volatile long*seqw=flag+32; long seen=*flag; long total=copies*n+PILOTS+1;
  *ready=1;                                                  /* armed: the sender may fire */
  double*ts=malloc((total+8)*8); long*dl=malloc((total+8)*8); long*sq=malloc((total+8)*8); long k=0; double tw=ns();
  while(k<total){ long f; while((f=*flag)==seen){ if(ns()-tw>timeout_s*1e9) goto done; } dl[k]=f-seen; seen=f; ts[k++]=ns(); sq[k-1]=*seqw; tw=ns(); if(sq[k-1]==-1) break; }
  done:;
  // decode by SEQUENCE: an edge is trusted only if the edge before it in the stream was its predecessor (no gap hides a miss)
  long*val=malloc((copies*n+1)*8); char*ok=calloc(copies*n+1,1);
  for(long j=1;j<k;j++){ long s2=sq[j], s1=sq[j-1]; if(s2<0) continue;
    int pred_ok = (s1==s2-1) || (s2==0 && s1==-2);        /* the edge before it must be its predecessor (or the last pilot for the first) */
    if(!pred_ok) continue;
    long angle=(long)floor((ts[j]-ts[j-1])/tick+0.5)-1-G; long st=dl[j]; if(labs(st)!=1 && labs(st)<1) continue; long sgn=st<0?-1:1; long laps=labs(st)-1;
    if(angle<0||angle>=DIAL) continue; val[s2]=sgn*(angle+DIAL*laps); ok[s2]=1; }
  int nmiss=0;
  for(int i=0;i<n;i++){ long v[8]; int m=0;
    for(int c=0;c<copies&&c<8;c++){ long s2=(long)c*n+i; if(ok[s2]) v[m++]=val[s2]; }
    if(m==0){ out[i]=0; if(missing && nmiss<maxmiss) missing[nmiss]=i; nmiss++; continue; }
    // median
    for(int a=0;a<m;a++) for(int b=a+1;b<m;b++) if(v[b]<v[a]){ long t2=v[a]; v[a]=v[b]; v[b]=t2; }
    out[i]=v[m/2]; }
  free(ts); free(dl); free(sq); free(val); free(ok); return missing? nmiss : (int)k; }

// repair: send the given (index, value) pairs; the receiver fills them by sequence
int edge_repair_send(const char*path, const long*idx, const long*vals, int m, double tick){
  volatile long*flag=line(path); volatile long*ready=flag+16; volatile long*seqw=flag+32;
  double tw=ns(); while(!*ready){ if(ns()-tw>60e9) return -1; } *ready=0; long e=*flag;
  double t=ns()+100000; for(int p=0;p<PILOTS;p++){ while(ns()<t+p*100*tick); *seqw=-2; *flag=++e; }
  double at=t+(PILOTS-1)*100*tick;
  for(int i=0;i<m;i++){ long v=vals[i]; long mm=labs(v); long angle=mm%DIAL; long step=(v<0?-1:1)*(1+mm/DIAL);
    at+=(angle+1+G)*tick; while(ns()<at); *seqw=idx[i]; e+=step; *flag=e; }
  usleep(500); *seqw=-1; *flag=++e; return m; }
// receive a repair: fills out[idx] for each edge that arrives with a valid predecessor; returns how many were filled
int edge_repair_recv(const char*path, long*out, const long*want, int m, double tick, double timeout_s){
  volatile long*flag=line(path); volatile long*ready=flag+16; volatile long*seqw=flag+32; long seen=*flag; long total=m+PILOTS+1;
  *ready=1; double*ts=malloc((total+8)*8); long*dl=malloc((total+8)*8); long*sq=malloc((total+8)*8); long k=0; double tw=ns();
  while(k<total){ long f; while((f=*flag)==seen){ if(ns()-tw>timeout_s*1e9) goto done; } dl[k]=f-seen; seen=f; ts[k++]=ns(); sq[k-1]=*seqw; tw=ns(); if(sq[k-1]==-1) break; }
  done:;
  int filled=0;
  for(long j=1;j<k;j++){ long s2=sq[j]; if(s2<0) continue; long angle=(long)floor((ts[j]-ts[j-1])/tick+0.5)-1-G; if(angle<0||angle>=DIAL) continue;
    long st=dl[j]; long sgn=st<0?-1:1; long laps=labs(st)-1; long v=sgn*(angle+DIAL*laps);
    // trust it only if the previous edge was the expected neighbour in this repair stream (pilot or previous index)
    int ok=0; for(int i=0;i<m;i++) if(want[i]==s2){ ok = (i==0)? (sq[j-1]==-2) : (sq[j-1]==want[i-1]); break; }
    if(ok){ out[s2]=v; filled++; } }
  free(ts); free(dl); free(sq); return filled; }
