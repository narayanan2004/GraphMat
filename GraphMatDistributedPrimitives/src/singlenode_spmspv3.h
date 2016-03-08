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


#ifndef SRC_SINGLENODE_SPMSPV3_H_
#define SRC_SINGLENODE_SPMSPV3_H_
#include <xmmintrin.h>
#include "src/bitvector.h"

template <typename Ta, typename Tx, typename Tvp, typename Ty>
void my_spmspv3(int* row_inds, int* col_ptrs, int* col_indices, Ta* vals,
               int num_partitions, int* row_pointers, int* col_starts,
               int* edge_pointers, Tx* xvalue, int * xbit_vector, 
	       Tvp * vpvalue, int * vpbit_vector, Ty * yvalue,
               int * ybit_vector, 
	       int m, int n, int* nnz, Ty (*op_mul)(Ta, Tx, Tvp),
               Ty (*op_add)(Ty, Ty)) {

#pragma omp parallel for schedule(dynamic, 1)
  for (int p = 0; p < num_partitions; p++) {
    // For each column
    const int* column_offset = col_indices + col_starts[p];
    const int* partitioned_row_offset = row_inds + edge_pointers[p];
    const Ta* partitioned_val_offset = vals + edge_pointers[p];
    const int* col_ptrs_cur = col_ptrs + col_starts[p];
    for (int j = 0; j < (col_starts[p + 1] - col_starts[p]) - 1; j++) {
      int col_index = col_indices[col_starts[p] + j];
      if(get_bitvector(col_index, xbit_vector)) {
        Tx Xval = xvalue[col_index];
	Tvp VPVal = vpvalue[col_index];
	assert(get_bitvector(col_index, vpbit_vector));
        _mm_prefetch((char*)(xvalue + column_offset[j + 4]), _MM_HINT_T0);

        int nz_idx = col_ptrs_cur[j];
        for (; nz_idx < col_ptrs_cur[j + 1]; nz_idx++) {
          int row_ind = partitioned_row_offset[nz_idx];
          Ta Aval = partitioned_val_offset[nz_idx];
          if(get_bitvector(row_ind, ybit_vector))
	  {
            yvalue[row_ind] = op_add(yvalue[row_ind], op_mul(Aval, Xval, VPVal));
	  }
	  else
	  {
            yvalue[row_ind] = op_mul(Aval, Xval, VPVal);
            set_bitvector(row_ind, ybit_vector);
	  }
        }
      }
    }
  }
  for (int p = 0; p < num_partitions; p++) {
    // nnz += new_nnz[p];
  }
  *nnz = m * n;
}

#endif  // SRC_SINGLENODE_SPMSPV3_H_
