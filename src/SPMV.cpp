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
#include <climits>
#include <cassert>

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>

template <class T, class U, class V, class E>
void Mulfn(E a, T b, U * c, void* gpv) { 
  GraphProgram<T,U,V,E>* gp = (GraphProgram<T,U,V,E>*) gpv;
  V dummy;
  gp->process_message(b, a, dummy, *c); 
}

template <class T, class U, class V, class E>
void Addfn(U a, U b, U * c, void* gpv) { 
  GraphProgram<T,U,V,E>* gp = (GraphProgram<T,U,V,E>*) gpv; 
  *c = a;
  gp->reduce_function(*c, b);
}


template <class T, class U, class V, class E>
void SpMSpV(const Graph<V,E>& G, const GraphProgram<T,U,V,E>* gp, const GraphPad::SpVec<GraphPad::DenseSegment<T> >& x, GraphPad::SpVec<GraphPad::DenseSegment<U> >& y) {
  struct timeval start, end;
  gettimeofday(&start, 0);

  GraphPad::SpMSpV(G.A, x, &y, Mulfn<T,U,V,E>, Addfn<T,U,V,E>, (void*)gp);

  #ifdef __TIMING
  gettimeofday(&end, 0);
  double time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
  printf("SPMSPV: time = %.3f ms \n", time);
  #endif
}
template <class T, class U, class V, class E>
void SpMTSpV(const Graph<V,E>& G, const GraphProgram<T,U,V,E>* gp, const GraphPad::SpVec<GraphPad::DenseSegment<T> >& x, GraphPad::SpVec<GraphPad::DenseSegment<U> >& y) {
  struct timeval start, end;
  gettimeofday(&start, 0);

  GraphPad::SpMSpV(G.AT, x, &y, Mulfn<T,U,V,E>, Addfn<T,U,V,E>, (void*)gp);

  #ifdef __TIMING
  gettimeofday(&end, 0);
  double time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
  printf("SPMTSPV: time = %.3f ms \n", time);
  #endif
}

