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

#include "src/graphpad.h"
//#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <vector>
#include <utility>
#include <sys/time.h>

#ifdef __ASSERT
#include <assert.h>
#endif

//#include "class_MatrixDC.cpp"
#include "Graph.cpp"
#include "GraphProgram.cpp"
//#include "SparseVector.cpp"
#include "SPMV.cpp"

int nthreads;

template<class T, class U, class V>
struct run_graph_program_temp_structure {
  //SparseInVector<T>* px;
  //SparseOutVector<U>* py;
};

template<class T, class U, class V, class E>
struct run_graph_program_temp_structure<T,U,V> graph_program_init(const GraphProgram<T,U,V,E>& gp, const Graph<V, E>& g) {

  struct run_graph_program_temp_structure<T,U,V> rgpts;
  //rgpts.px = new SparseInVector<T>(g.nvertices);
  //rgpts.py = new SparseOutVector<U>(g.nvertices);
  return rgpts;
}

template<class T, class U, class V>
void graph_program_clear(struct run_graph_program_temp_structure<T,U,V>& rgpts) {
  //delete rgpts.px;
  //delete rgpts.py;
}
/*
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
*/


template <class T,class U, class V, class E>
void send_message(bool a, V _v, T* b, void* gpv) {
  GraphProgram<T,U,V,E>* gp = (GraphProgram<T,U,V,E>*) gpv;
  if(a == true) {
    gp->send_message(_v, *b);
  }
}

template <class T, class U, class V, class E>
void apply_func(U y, V* b, void* gpv) {
  GraphProgram<T,U,V,E>* gp = (GraphProgram<T,U,V,E>*) gpv;
  gp->apply(y, *b);
}

template <class T, typename U, class V, class E>
void run_graph_program(GraphProgram<T,U,V,E>* gp, Graph<V,E>& g, int iterations=1, struct run_graph_program_temp_structure<T,U,V>* rgpts=NULL) { //iterations = -1 ==> until convergence
  int it = 0;
  int converged = 1;

  struct timeval start, end, init_start, init_end, iteration_start, iteration_end;
  double time;
  
  //unsigned long long int init_start = __rdtsc();
  gettimeofday(&init_start, 0);
  

  auto act = gp->getActivity();

  //SparseInVector<T>* px;
  //SparseOutVector<U>* py;
  GraphPad::SpVec<GraphPad::DenseSegment<T> >* px;
  GraphPad::SpVec<GraphPad::DenseSegment<U> >* py;

  if (rgpts == NULL) {
    px  = new GraphPad::SpVec<GraphPad::DenseSegment<T> >();
    px->AllocatePartitioned(g.nvertices, GraphPad::global_nrank, GraphPad::vector_partition_fn);
    T _t;
    px->setAll(_t);
    py  = new GraphPad::SpVec<GraphPad::DenseSegment<U> >();
    py->AllocatePartitioned(g.nvertices, GraphPad::global_nrank, GraphPad::vector_partition_fn);
    U _u;
    py->setAll(_u);
  }

  GraphPad::SpVec<GraphPad::DenseSegment<T> >& x = *px;
  GraphPad::SpVec<GraphPad::DenseSegment<T> >& y = *py;
  //SparseInVector<T>&x = (rgpts==NULL)?(*px):*(rgpts->px);
  //SparseOutVector<U>& y = (rgpts==NULL)?(*py):*(rgpts->py);

  #ifdef __TIMING
  printf("Nvertices = %d \n", g.getNumberOfVertices());
  #endif

  //unsigned long long int start, end;

  //unsigned long long int init_end = __rdtsc();
  gettimeofday(&init_end, 0);

  #ifdef __TIMING
  //printf("GraphMat init time = %f ms \n", (init_end-init_start)/(CPU_FREQ)*1e3);
  time = (init_end.tv_sec-init_start.tv_sec)*1e3+(init_end.tv_usec-init_start.tv_usec)*1e-3;  
  printf("GraphMat init time = %f ms \n", time);
  #endif

  while(1) {
    //unsigned long long int iteration_start = __rdtsc();
    gettimeofday(&iteration_start, 0);

    //x.clear();
    //y.clear();
    GraphPad::Clear(&x);
    GraphPad::Clear(&y);
    converged = 1;

    //start = __rdtsc();
    gettimeofday(&start, 0);


    //check active vector and set message vector
    /*int count = 0;
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
    x.length = count;*/
    GraphPad::IntersectReduce(g.active, g.vertexproperty, &x, send_message<T,U,V,E>, (void*)gp);
    //GraphPad::IntersectReduce(g.active, g.vertexproperty, &x, send_message<int, int, V>, (void*)gp);
    //y.setAll(0);

    #ifdef __TIMING
    printf("x.length = %d \n", x.getNNZ());
    #endif
    //end = __rdtsc();
    gettimeofday(&end, 0);
    #ifdef __TIMING
    time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
    //printf("Send message time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
    printf("Send message time = %.3f ms \n", time);
    #endif

    //start = __rdtsc();
    gettimeofday(&start, 0);

    
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
    //end = __rdtsc();
    gettimeofday(&end, 0);
    #ifdef __TIMING
    //printf("SPMV time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
    time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
    printf("SPMV time = %.3f ms \n", time);
    #endif
    
    //start = __rdtsc();
    gettimeofday(&start, 0);
    g.setAllInactive();

    //printf("y[1] = %d\n", py->get(1));

    //GraphPad::SpVec<GraphPad::DenseSegment<U> > tmp;
    //U tmp_v; tmp.get(1, &tmp_v);
    //U tmp_v; y.get(1, &tmp_v);
    //std::cout<< "y[1] = " << tmp_v << std::endl;


    //update state and activity and check for convergence if needed
    int nout = 0;
    int total_search = 0;
    int local_converged = 1;
    converged = 1;

    //GraphPad::IntersectReduce(g.active, y, &g.vertexproperty, set_y<U,V>);
    //auto apply_func  = set_y_apply<U,V>;
    //GraphPad::Apply(y, &g.vertexproperty, apply_func<T,U,V>, (void*)gp);
    for(int segmentId = 0 ; segmentId < y.nsegments ; segmentId++)
    {
      if(y.nodeIds[segmentId] == GraphPad::global_myrank)
      {
        auto segment = y.segments[segmentId].properties;
        auto vpValueArray = g.vertexproperty.segments[segmentId].properties.value;
        #pragma omp parallel for reduction(&:local_converged) 
        for (int i = 0; i < y.segments[segmentId].num_ints; i++) {
          unsigned int value = segment.bit_vector[i];
          while (value != 0) {
            int last_bit = _bit_scan_forward(value);
            int idx = i*32 + last_bit; 

            V old_prop;
            //old_prop = g.vertexproperty.segments[segmentId].properties.value[idx];
            old_prop = vpValueArray[idx];
      
            //gp->apply(segment.value[idx], g.vertexproperty.segments[segmentId].properties.value[idx]);
            gp->apply(segment.value[idx], vpValueArray[idx]);
            if (old_prop != vpValueArray[idx]) {
              //g.active.segments[segmentId].set(idx+1, true);
              g.active.segments[segmentId].properties.value[idx] = true;
              GraphPad::set_bitvector(idx, g.active.segments[segmentId].properties.bit_vector);
              local_converged = 0;
            }

            value &= (~(1<<last_bit));
          }
        }
        
      }
    }
    MPI_Allreduce(&local_converged, &converged, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    /*#pragma omp parallel num_threads(nthreads) reduction(+:nout) reduction(&:converged) reduction(+:total_search) //schedule(static)
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
    
    }*/
    if (act == ALL_VERTICES) {
      g.setAllActive();
    }


    //end = __rdtsc();
    gettimeofday(&end, 0);
    #ifdef __TIMING
    //printf("Apply time = %.3f ms \n", (end-start)/(CPU_FREQ)*1e3);
    time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
    printf("Apply time = %.3f ms \n", time);
    #endif
    
    gp->do_every_iteration(it);

    //unsigned long long int iteration_end = __rdtsc();
    gettimeofday(&iteration_end, 0);
    #ifdef __TIMING
    //printf("Iteration %d :: %f msec :: updated %d vertices \n", it, (iteration_end-iteration_start)/(CPU_FREQ)*1e3, y.getNNZ());
    time = (iteration_end.tv_sec-iteration_start.tv_sec)*1e3+(iteration_end.tv_usec-iteration_start.tv_usec)*1e-3;
    //printf("Number of vertices that changed state = %d \n", g.active.getNNZ());
    printf("Iteration %d :: %f msec :: updated %d vertices :: changed %d vertices \n", it, time, y.getNNZ(), g.active.getNNZ());
    #endif

    it++;
    if (it == iterations) {
      break;
    }
    if (iterations <= 0 && converged == 1) {
      break;
    }
  }

  //unsigned long long int clear_start = __rdtsc();
  struct timeval clear_start, clear_end;
  gettimeofday(&clear_start, 0);

  if (rgpts == NULL) {
    delete px;
    delete py;
  }

  //unsigned long long int clear_end = __rdtsc();
  gettimeofday(&clear_end, 0);
  #ifdef __TIMING
  //printf("GraphMat clear time = %f msec \n", (clear_end-clear_start)/(CPU_FREQ)*1e3);
  time = (clear_end.tv_sec-clear_start.tv_sec)*1e3+(clear_end.tv_usec-clear_start.tv_usec)*1e-3;
  printf("GraphMat clear time = %f msec \n", time);
  #endif

  printf("Completed %d iterations \n", it);

}

