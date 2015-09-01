/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ******************************************************************************/
/* Narayanan Sundaram (Intel Corp.)
 * ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <vector>
#include <utility>
#include <sys/time.h>

#ifdef __ASSERT
#include <assert.h>
#endif

#include "class_MatrixDC.cpp"
#include "Graph.cpp"
#include "GraphProgram.cpp"
#include "SparseVector.cpp"
#include "SPMV.cpp"

int nthreads;

template<class T, class U, class V>
struct run_graph_program_temp_structure {
  SparseInVector<T>* px;
  SparseOutVector<U>* py;
};

template<class T, class U, class V>
struct run_graph_program_temp_structure<T,U,V> graph_program_init(const GraphProgram<T,U,V>& gp, const Graph<V>& g) {

  struct run_graph_program_temp_structure<T,U,V> rgpts;
  rgpts.px = new SparseInVector<T>(g.nvertices);
  rgpts.py = new SparseOutVector<U>(g.nvertices);
  return rgpts;
}

template<class T, class U, class V>
void graph_program_clear(struct run_graph_program_temp_structure<T,U,V>& rgpts) {
  delete rgpts.px;
  delete rgpts.py;
}

template <class T, class U, class V>
void run_graph_program(GraphProgram<T,U,V>* gp, Graph<V>& g, int iterations=1, struct run_graph_program_temp_structure<T,U,V>* rgpts=NULL) { //iterations = -1 ==> until convergence
  int it = 0;
  int converged = 1;

  unsigned long long int init_start = __rdtsc();

  auto act = gp->getActivity();

  SparseInVector<T>* px;
  SparseOutVector<U>* py;

  if (rgpts == NULL) {
    px  = new SparseInVector<T>(g.nvertices);
    py  = new SparseOutVector<U>(g.nvertices);
  }

  SparseInVector<T>&x = (rgpts==NULL)?(*px):*(rgpts->px);
  SparseOutVector<U>& y = (rgpts==NULL)?(*py):*(rgpts->py);

  #ifdef __TIMING
  printf("Nvertices = %d numints = %d \n", g.nvertices, y.numInts);
  #endif

  unsigned long long int start, end;
  int* start_vertex = new int[nthreads+1];

  //divide numInts to start_vertex
  //divide the active vertices in each into start_index
  start_vertex[nthreads] = g.nvertices;
  #pragma omp parallel num_threads(nthreads)
  {
    int tid = omp_get_thread_num();
    int ints_per_th = (y.numInts/nthreads)*32;
    int sv  = ints_per_th*tid;
    sv = (((sv/32)/4)*4)*32; //sv is multiple of 32 and sv/32 is a multiple of 4
    sv = (((sv/32)/SIMD_WIDTH)*SIMD_WIDTH)*32; //sv is multiple of 32 and sv/32 is a multiple of SIMD_WIDTH
    if (sv >= g.nvertices) sv = g.nvertices;
    if (sv == 0) sv = 0;
    start_vertex[tid] = sv;
  }

  unsigned long long int init_end = __rdtsc();
  #ifdef __TIMING
  printf("GraphMat init time = %f ms \n", (init_end-init_start)/(CPU_FREQ)*1e3);
  #endif

  while(1) {
    unsigned long long int iteration_start = __rdtsc();
    x.clear();
    y.clear();
    converged = 1;

    start = __rdtsc();

    //check active vector and set message vector
    int count = 0;
    #pragma omp parallel num_threads(nthreads) reduction(+:count)
    {
    int tid = omp_get_thread_num();
    for (int i = start_vertex[tid]; i < start_vertex[tid+1]; i++){
      if (g.active[i]) {
        T message;
        bool msg_opt = gp->send_message(g.vertexproperty[i], message);
        if (msg_opt) {
          x.set(i, message);
          count++;
        }
      }
    }
    }
    x.length = count;

    #ifdef __TIMING
    printf("x.length = %d \n", x.length);
    #endif
    end = __rdtsc();
    #ifdef __TIMING
    printf("Send message time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
    #endif

    start = __rdtsc();
    
    //do SpMV
    if (gp->getOrder() == OUT_EDGES) {

      SpMTSpV(g, gp, x, y);

    } else if (gp->getOrder() == IN_EDGES) {

      SpMSpV(g, gp, x, y);

    } else if (gp->getOrder() == ALL_EDGES) {

      SpMTSpV(g, gp, x, y);
      SpMSpV(g, gp, x, y);

    } else {
      printf("Unrecognized option \n");
      exit(1);
    }
    end = __rdtsc();
    #ifdef __TIMING
    printf("SPMV time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
    #endif
    
    start = __rdtsc();
    g.setAllInactive();

    //update state and activity and check for convergence if needed
    int nout = 0;
    int total_search = 0;
    converged = 1;
    #pragma omp parallel num_threads(nthreads) reduction(+:nout) reduction(&:converged) reduction(+:total_search) //schedule(static)
    {
      int zero = 0;
      SIMDINTTYPE xmm_zero = _MM_SET1(zero);
      int tid = omp_get_thread_num();
      int count_ones = 0;
    int end_of_numInts = start_vertex[tid+1]/32;
    if (tid == nthreads-1) end_of_numInts = y.numInts;
    for (int ii = start_vertex[tid]/32; ii < end_of_numInts; ii+=SIMD_WIDTH) {

      __m128i xmm_local_bitvec = _mm_loadu_si128((__m128i*)(y.bitvector + ii));
      __m128 xmm_cmp_mask = _mm_castsi128_ps(_mm_cmpeq_epi32((xmm_local_bitvec), (xmm_zero)));
      int mask_value_0 = _mm_movemask_ps(xmm_cmp_mask);
      if(mask_value_0 == 15)
      {
        continue;
      }
      for(int i = ii; i < ii+SIMD_WIDTH; i++)
      {
        unsigned int value = y.bitvector[i];
        while (value != 0) {
          int last_bit = _bit_scan_forward(value);
          int idx = i*32 + last_bit;

          V old_prop;
            old_prop = g.vertexproperty[idx];
      
          gp->apply(y.value[idx], g.vertexproperty[idx]);
          nout++;

            if (old_prop != g.vertexproperty[idx]) {
	      g.setActive(idx);
              count_ones++;
              converged = 0;
              total_search++;
            }

          value &= (~(1<<last_bit));
        }
      }
    }
    
    }
    if (act == ALL_VERTICES) {
      g.setAllActive();
    }

    #ifdef __TIMING
    printf("Number of vertices that changed state = %d \n", total_search);
    #endif

    end = __rdtsc();
    #ifdef __TIMING
    printf("Apply time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
    #endif
    
    gp->do_every_iteration(it);

    unsigned long long int iteration_end = __rdtsc();
    #ifdef __TIMING
    printf("Iteration %d :: %f msec :: updated %d vertices \n", it, (iteration_end-iteration_start)/(CPU_FREQ)*1e3, nout);
    #endif

    it++;
    if (it == iterations) {
      break;
    }
    if (iterations <= 0 && converged == 1) {
      break;
    }
  }

  unsigned long long int clear_start = __rdtsc();
  delete [] start_vertex;

  if (rgpts == NULL) {
    delete px;
    delete py;
  }

  unsigned long long int clear_end = __rdtsc();
  #ifdef __TIMING
  printf("GraphMat clear time = %f msec \n", (clear_end-clear_start)/(CPU_FREQ)*1e3);
  #endif

  printf("Completed %d iterations \n", it);

}




