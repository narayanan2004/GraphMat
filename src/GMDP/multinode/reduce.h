/******************************************************************************
 * ** Copyright (c) 2016, Intel Corporation                                     **
 * ** All rights reserved.                                                      **
 * **                                                                           **
 * ** Redistribution and use in source and binary forms, with or without        **
 * ** modification, are permitted provided that the following conditions        **
 * ** are met:                                                                  **
 * ** 1. Redistributions of source code must retain the above copyright         **
 * **    notice, this list of conditions and the following disclaimer.          **
 * ** 2. Redistributions in binary form must reproduce the above copyright      **
 * **    notice, this list of conditions and the following disclaimer in the    **
 * **    documentation and/or other materials provided with the distribution.   **
 * ** 3. Neither the name of the copyright holder nor the names of its          **
 * **    contributors may be used to endorse or promote products derived        **
 * **    from this software without specific prior written permission.          **
 * **                                                                           **
 * ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
 * ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
 * ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
 * ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
 * ** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
 * ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
 * ** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
 * ** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
 * ** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
 * ** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
 * ** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * ******************************************************************************/
/* Michael Anderson (Intel Corp.)
 *  * ******************************************************************************/


#ifndef SRC_MULTINODE_REDUCE_H_
#define SRC_MULTINODE_REDUCE_H_

#include "GMDP/matrices/SpMat.h"
#include "GMDP/singlenode/reduce.h"

template <template<typename> class SpSegment, typename T, typename VT>
void MapReduce(SpVec<SpSegment<VT> > * vec, T* res,
                 void (*op_map)(VT*, T*, void*), void (*op_fp)(T, T, T*, void*), void* vsp) {
  int global_myrank = get_global_myrank();
  int global_nrank = get_global_nrank();
  int start = 0;
  int end = vec->nsegments;
  bool res_set = false;

  for (int i = start; i < end; i++) {
    if (vec->nodeIds[i] == global_myrank) {
      mapreduce_segment(vec->segments[i], res, &res_set, op_map, op_fp, vsp);
    }
  }

  // Reduce across nodes
  T* all_res = new T[global_nrank];
  all_res[global_myrank] = *res;
  MPI_Status status;
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == i) {
      MPI_Send(all_res + i, sizeof(T), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
  }
  for (int i = 1; i < global_nrank; i++) {
    if (global_myrank == 0) {
      MPI_Recv(all_res + i, sizeof(T), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
      T tmp_res = *res;
      op_fp(tmp_res, all_res[i], res, vsp);
    }
  }

  // Broadcast to all nodes
  MPI_Bcast(res, sizeof(T), MPI_CHAR, 0, MPI_COMM_WORLD);
  delete [] all_res;
}


#endif  // SRC_MULTINODE_REDUCE_H_
