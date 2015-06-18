#include <climits>
#include <cassert>

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>

extern int nthreads;

template <class T, class U, class V, class E>
void BlockingHypersparse_GEMV(const MatrixDC<E>* A_DCSC, const V* Vertexproperty, const SparseInVector<T>& x, SparseOutVector<U>& result, const GraphProgram<T, U, V>* gp) {
  int i = 0;
  if (A_DCSC->nnz == 0) return;

    i = 0;
    U res[4];

    while (i < A_DCSC->nzx) {
      if (get_bitvector(A_DCSC->xindex[i], x.bitvector)) {
        int end = (i == A_DCSC->nzx - 1 ? A_DCSC->nnz : A_DCSC->starty[i + 1]);
        T  p = x.getValue(A_DCSC->xindex[i]);
	int j = A_DCSC->starty[i];
        _mm_prefetch((char*)(x.value + A_DCSC->xindex[i+4]), _MM_HINT_T0);
        for (; j < end-3; j+=4) {
          gp->process_message(p, A_DCSC->value[j+0], Vertexproperty[A_DCSC->yindex[j+0]], res[0]);
          gp->process_message(p, A_DCSC->value[j+1], Vertexproperty[A_DCSC->yindex[j+1]], res[1]);
          gp->process_message(p, A_DCSC->value[j+2], Vertexproperty[A_DCSC->yindex[j+2]], res[2]);
          gp->process_message(p, A_DCSC->value[j+3], Vertexproperty[A_DCSC->yindex[j+3]], res[3]);
          result.reduce(A_DCSC->yindex[j+0], res[0], gp); 
          result.reduce(A_DCSC->yindex[j+1], res[1], gp); 
          result.reduce(A_DCSC->yindex[j+2], res[2], gp); 
          result.reduce(A_DCSC->yindex[j+3], res[3], gp); 
        }
        for (; j < end; j++) {
          gp->process_message(p, A_DCSC->value[j], Vertexproperty[A_DCSC->yindex[j]], res[0]);
          result.reduce(A_DCSC->yindex[j], res[0], gp); 
        }
      }
      i++;
    }
}



template <class T, class U, class V>
void SpMSpV(const Graph<V>& G, const GraphProgram<T,U,V>* gp, const SparseInVector<T>& x, SparseOutVector<U>& y) {
    unsigned long long int start = __rdtsc();
    #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
    for (int i = 0; i < G.nparts; i++) { // loop over blocks
      BlockingHypersparse_GEMV(G.mat[i], G.vertexproperty, x, y, gp);
    }
    unsigned long long int end = __rdtsc();
    #ifdef __TIMING
    printf("SPMSPV: time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
    #endif
}

template <class T, class U, class V>
void SpMTSpV(const Graph<V>& G, const GraphProgram<T,U,V>* gp, const SparseInVector<T>& x, SparseOutVector<U>& y) {

  unsigned long long int start = __rdtsc();
  #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 1)
  for (int i = 0; i < G.nparts; i++) { // loop over blocks
      BlockingHypersparse_GEMV(G.matT[i], G.vertexproperty, x, y, gp); 
  }
  unsigned long long int end = __rdtsc();
  #ifdef __TIMING
  printf("SPMTSPV: time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
  #endif
}


