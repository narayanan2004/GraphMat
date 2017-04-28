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


#pragma once

#include <xmmintrin.h>
#include "GMDP/utils/bitvector.h"

template <typename Ta, typename Tvp>
void my_applyedges(int* row_inds, int* col_ptrs, int* col_indices, Ta* vals,
               int num_partitions, int* row_pointers, int* col_starts,
               int* edge_pointers, Tvp * vpvalue1, int * vpbit_vector1,
	       Tvp * vpvalue2, int * vpbit_vector2,
	       int m, int n, int* nnz, 
               void (*op)(Ta*, Tvp, Tvp, void*), void* vsp) {

#pragma omp parallel for schedule(dynamic, 1)
  for (int p = 0; p < num_partitions; p++) {
    // For each column
    const int* column_offset = col_indices + col_starts[p];
    const int* partitioned_row_offset = row_inds + edge_pointers[p];
    Ta* partitioned_val_offset = vals + edge_pointers[p];
    const int* col_ptrs_cur = col_ptrs + col_starts[p];
    for (int j = 0; j < (col_starts[p + 1] - col_starts[p]) - 1; j++) {
      int col_index = col_indices[col_starts[p] + j];
      {
	      //assert(get_bitvector(col_index, vpbit_vector1));
        Tvp Xval = vpvalue1[col_index];
        _mm_prefetch((char*)(vpvalue1 + column_offset[j + 4]), _MM_HINT_T0);

        int nz_idx = col_ptrs_cur[j];
        for (; nz_idx < col_ptrs_cur[j + 1]; nz_idx++) {
          int row_ind = partitioned_row_offset[nz_idx];
	        assert(get_bitvector(row_ind, vpbit_vector2));
          //Tvp Yval = vpvalue2[row_ind];
          //Ta Aval = partitioned_val_offset[nz_idx];
          //op(&Aval, Xval, Yval, vsp);
          op(&partitioned_val_offset[nz_idx], Xval, vpvalue2[row_ind], vsp);
          //partitioned_val_offset[nz_idx] = Aval;
        }
      }
    }
  }
  for (int p = 0; p < num_partitions; p++) {
    // nnz += new_nnz[p];
  }
  *nnz = m * n;
}


template <typename Ta, typename Tvp>
void apply_edges(DCSCTile<Ta>* tile, const DenseSegment<Tvp> * segmentvp1,
                  const DenseSegment<Tvp> * segmentvp2,
                  void (*fp)(Ta*, Tvp, Tvp, void*), void* vsp) {
  if(!(tile->isEmpty()))
  {
  int nnz;
  my_applyedges(tile->row_inds, tile->col_ptrs, tile->col_indices, tile->vals,
            tile->num_partitions, tile->row_pointers, tile->col_starts,
            tile->edge_pointers, segmentvp1->properties->value, segmentvp1->properties->bit_vector,
     	    segmentvp2->properties->value, segmentvp2->properties->bit_vector,
            tile->m, tile->n, (&nnz),
            fp, vsp);
  }
}

