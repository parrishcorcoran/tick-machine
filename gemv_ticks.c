// The same dot product twice. Once reading fp32 weights (4 bytes each), once
// reading int16 ticks (2 bytes each) with the dial table in cache.
// If moving weights is the cost, the tick loop runs ~2x faster. Batch 1, GEMV.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
    int n_in=768, n_out=3072, dial=4096, reps=200;
    int layers = argc>1 ? atoi(argv[1]) : 48;          // GPT-2 has 48 linear layers; sweep so weights exceed cache
    size_t n=(size_t)n_in*n_out*layers;
    float *W=malloc(n*4); int16_t *T=malloc(n*2); float *table=malloc(dial*4);
    float *x=malloc(n_in*4), *y=malloc(n_out*4);
    int half=dial/2-1;
    for(int t=0;t<dial;t++){int s=((t+dial/2)%dial)-dial/2; table[t]=(float)s/half;}
    srand(1);
    for(size_t i=0;i<n;i++){int q=rand()%(2*half+1)-half; T[i]=(int16_t)((q+dial)%dial); W[i]=table[T[i]];}
    for(int i=0;i<n_in;i++) x[i]=(float)rand()/RAND_MAX-0.5f;
    double best_f=1e9,best_t=1e9; double chk_f=0,chk_t=0;
    for(int r=0;r<reps/layers+1;r++){
        double t0=now();
        for(int l=0;l<layers;l++){ const float*Wl=W+(size_t)l*n_in*n_out;
            for(int j=0;j<n_out;j++){float s=0; const float*w=Wl+(size_t)j*n_in; for(int i=0;i<n_in;i++) s+=w[i]*x[i]; y[j]=s;}
            chk_f+=y[0];}
        double d=(now()-t0)/layers; if(d<best_f)best_f=d;
        t0=now();
        for(int l=0;l<layers;l++){ const int16_t*Tl=T+(size_t)l*n_in*n_out;
            for(int j=0;j<n_out;j++){float s=0; const int16_t*w=Tl+(size_t)j*n_in; for(int i=0;i<n_in;i++) s+=table[w[i]]*x[i]; y[j]=s;}
            chk_t+=y[0];}
        d=(now()-t0)/layers; if(d<best_t)best_t=d;
    }
    double bytes_f=(double)n_in*n_out*4, bytes_t=(double)n_in*n_out*2;
    printf("%d layers (%.0f MB fp32 / %.0f MB ticks), 768x3072 GEMV per layer, batch 1\n", layers, n*4/1e6, n*2/1e6);
    printf("  fp32 weights : %7.3f ms/layer  %6.1f GB/s\n", best_f*1e3, bytes_f/best_f/1e9);
    printf("  int16 ticks  : %7.3f ms/layer  %6.1f GB/s   -> %.2fx faster, same answer: %s\n",
           best_t*1e3, bytes_t/best_t/1e9, best_f/best_t, fabs(chk_f-chk_t)<1e-3*fabs(chk_f)+1e-6?"yes":"NO");
    return 0;
}
