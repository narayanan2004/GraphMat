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
#include "GMDP/utils/bitvector.h"

template <typename Ta, typename Tx, typename Tvp, typename Ty>
void my_spmspv3(int* row_inds, int* col_ptrs, int* col_indices, Ta* vals,
               int num_partitions, int* row_pointers, int* col_starts,
               int* edge_pointers, Tx* xvalue, int * xbit_vector, 
	       Tvp * vpvalue, int * vpbit_vector, Ty * yvalue,
               int * ybit_vector, 
	       int m, int n, int* nnz, void (*op_mul)(Ta, Tx, Tvp, Ty*, void*),
               void (*op_add)(Ty, Ty, Ty*, void*), void* vsp) {

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
        _mm_prefetch((char*)(xvalue + column_offset[j + 4]), _MM_HINT_T0);

        int nz_idx = col_ptrs_cur[j];
        for (; nz_idx < col_ptrs_cur[j + 1]; nz_idx++) {
          int row_ind = partitioned_row_offset[nz_idx];
	  //Tvp VPVal = vpvalue[row_ind];
	  assert(get_bitvector(row_ind, vpbit_vector));
          Ta Aval = partitioned_val_offset[nz_idx];
          if(get_bitvector(row_ind, ybit_vector))
	  {
            Ty tmp_mul;
            //Ty tmp_add;
            op_mul(Aval, Xval, vpvalue[row_ind], &tmp_mul, vsp);
            op_add(yvalue[row_ind], tmp_mul, &yvalue[row_ind], vsp);
            //yvalue[row_ind] = tmp_add;
	  }
	  else
	  {
            //Ty tmp_mul;
            //op_mul(Aval, Xval, VPVal, &tmp_mul, vsp);
            op_mul(Aval, Xval, vpvalue[row_ind], &yvalue[row_ind], vsp);
            //yvalue[row_ind] = tmp_mul;
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


template <typename Ta, typename Tx, typename Tvp, typename Ty>
void my_coospmspv3(Ta* a, int* ia, int* ja, int num_partitions, int * partition_starts,
                  Tx* xvalue, int * xbit_vector,
	          Tvp * vpvalue, int * vpbit_vector, Ty * yvalue,
                  int * ybit_vector, 
 	          int m, int n, int* nnz, void (*op_mul)(Ta, Tx, Tvp, Ty*, void*),
                  void (*op_add)(Ty, Ty, Ty*, void*), void* vsp) {



  #pragma omp parallel for schedule(dynamic, 1)
  for(int partition = 0 ; partition < num_partitions ; partition++)
  {
    for(int nz = partition_starts[partition] ; nz < partition_starts[partition+1] ; nz++)
    {
      int row = ia[nz]-1;
      int col = ja[nz]-1;
#ifdef __DEBUG
      assert(row < m);
      assert(row >= 0);
      assert(col < n);
      assert(col >= 0);
#endif
      if(get_bitvector(col, xbit_vector))
      {
	Tvp VPVal = vpvalue[row];
	assert(get_bitvector(row, vpbit_vector));

        Ty tmp_mul;
        op_mul(a[nz], xvalue[col], VPVal, &tmp_mul, vsp);

        bool row_exists = get_bitvector(row, ybit_vector);
        if(!row_exists)
        {
          yvalue[row] = tmp_mul;
        }
        else
        {
          Ty tmp_add = yvalue[row];
          Ty yval;
          op_add(tmp_add, tmp_mul, &yval, vsp);
          yvalue[row] = yval;
        }
        set_bitvector(row, ybit_vector);
      }
    }
  }
}


template <typename Ta, typename Tx, typename Tvp, typename Ty>
void mult_segment3(const DCSCTile<Ta>* tile, const DenseSegment<Tx> * segmentx,
                  const DenseSegment<Tvp> * segmentvp,
                  DenseSegment<Ty> * segmenty,
                  void (*mul_fp)(Ta, Tx, Tvp, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  segmenty->alloc();
  segmenty->initialize();
  int nnz = 0;
  my_spmspv3(tile->row_inds, tile->col_ptrs, tile->col_indices, tile->vals,
            tile->num_partitions, tile->row_pointers, tile->col_starts,
            tile->edge_pointers, segmentx->properties->value, segmentx->properties->bit_vector,
     	    segmentvp->properties->value, segmentvp->properties->bit_vector,
            segmenty->properties->value, segmenty->properties->bit_vector, tile->m, tile->n, (&nnz),
            mul_fp, add_fp, vsp);
  segmenty->properties->nnz = segmenty->compute_nnz();
}

template <typename Ta, typename Tx, typename Tvp, typename Ty>
void mult_segment3(const COOTile<Ta>* tile, const DenseSegment<Tx> * segmentx,
                  const DenseSegment<Tvp> * segmentvp,
                  DenseSegment<Ty>* segmenty,
                  void (*mul_fp)(Ta, Tx, Tvp, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp) {
  segmenty->alloc();
  segmenty->initialize();
  int nnz = 0;
  my_coospmspv3(tile->a, tile->ia, tile->ja, tile->num_partitions, tile->partition_start,
               segmentx->properties->value, segmentx->properties->bit_vector,
     	       segmentvp->properties->value, segmentvp->properties->bit_vector,
               segmenty->properties->value, segmenty->properties->bit_vector, tile->m, tile->n, (&nnz),
               mul_fp, add_fp, vsp);
  segmenty->properties->nnz = segmenty->compute_nnz();
}




#endif  // SRC_SINGLENODE_SPMSPV3_H_
