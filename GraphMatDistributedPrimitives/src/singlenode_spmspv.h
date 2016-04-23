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
/* Narayanan Sundaram (Intel Corp.), Michael Anderson (Intel Corp.)
 *  * ******************************************************************************/


#ifndef SRC_SINGLENODE_SPMSPV_H_
#define SRC_SINGLENODE_SPMSPV_H_
#include <xmmintrin.h>
#include "src/bitvector.h"


template <typename Ta, typename Tx, typename Ty>
void my_spmspv(int* row_inds, int* col_ptrs, int* col_indices, Ta* vals,
               int num_partitions, int* row_pointers, int* col_starts,
               int* edge_pointers, Tx* xvalue, int * xbit_vector, Ty* yvalue,
               int * ybit_vector, int m, int n, int* nnz, void (*op_mul)(Ta, Tx, Ty*, void*),
               void (*op_add)(Ty, Ty, Ty*, void*), void* vsp) {

// int * new_nnz = new int[num_partitions];
// memset(new_nnz, 0, num_partitions * sizeof(int));
#pragma omp parallel for schedule(dynamic, 1)
  for (int p = 0; p < num_partitions; p++) {
    // For each column
    const int* column_offset = col_indices + col_starts[p];
    const int* partitioned_row_offset = row_inds + edge_pointers[p];
    const Ta* partitioned_val_offset = vals + edge_pointers[p];
    const int* col_ptrs_cur = col_ptrs + col_starts[p];
    for (int j = 0; j < (col_starts[p + 1] - col_starts[p]) - 1 ; j++) {
      int col_index = col_indices[col_starts[p] + j];
      if(get_bitvector(col_index, xbit_vector)) {
        Tx Xval = xvalue[col_index];
        _mm_prefetch((char*)(xvalue + column_offset[j + 4]), _MM_HINT_T0);

        int nz_idx = col_ptrs_cur[j];
        for (; nz_idx < col_ptrs_cur[j + 1]; nz_idx++) {
          int row_ind = partitioned_row_offset[nz_idx];
          Ta Aval = partitioned_val_offset[nz_idx];
	  Ty temp_mul_result;
          op_mul(Aval, Xval, &temp_mul_result, vsp);
          if(get_bitvector(row_ind, ybit_vector))
	  {
	    Ty temp_y_copy = yvalue[row_ind];
            op_add(temp_y_copy, temp_mul_result, &(yvalue[row_ind]), vsp);
	  }
	  else
	  {
            yvalue[row_ind] = temp_mul_result;
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

template <typename Ta, typename Tx, typename Ty>
void my_csrspmspv(Ta* a, int* ia, int* ja, Tx* xvalue, int * xbit_vector,
                  Ty* yvalue, int * ybit_vector, int m, int n, int* nnz,
                  void (*op_mul)(Ta, Tx, Ty*, void*), void (*op_add)(Ty, Ty, Ty*, void*), void* vsp) {

  int num_partitions = omp_get_max_threads() * 4;
  int rows_per_partition = (m + num_partitions - 1) / num_partitions;
  rows_per_partition = ((rows_per_partition + 31) / 32) * 32;

  #pragma omp parallel for schedule(dynamic, 1)
  for(int partition = 0 ; partition < num_partitions ; partition++)
  {
    int start_row = partition * rows_per_partition;
    int end_row = (partition+1) * rows_per_partition;
    if(end_row > m) end_row = m;
    for(int row = start_row ; row < end_row ; row++)
    {
      bool row_exists = get_bitvector(row, ybit_vector);
      Ty yval;
      if(row_exists)
      {
        yval = yvalue[row];
      }
      for (int nz = ia[row]; nz < ia[row + 1]; nz++) {
        Ty tmp_mul;
        int col_id = ja[nz-1]-1;
        if(get_bitvector(col_id, xbit_vector))
        {
          op_mul(a[nz - 1], xvalue[col_id], &tmp_mul, vsp);
          if(row_exists)
          {
            Ty tmp_add = yval;
            op_add(tmp_add, tmp_mul, &yval, vsp);
          } 
          else
          {
            yval = tmp_mul;
            set_bitvector(row, ybit_vector);
  	    row_exists=true;
          }
        }
      }
      if(row_exists)
      {
        yvalue[row] = yval;
      }
    }
  }
  //*nnz = m * n;
}


template <typename Ta, typename Tx, typename Ty>
void my_dcsrspmspv(Ta* a, int* ia, int* ja, int * row_ids, int num_rows, int * partition_ptrs, int num_partitions,
                  Tx* xvalue, int * xbit_vector,
                  Ty* yvalue, int * ybit_vector, int m, int n, int* nnz,
                  void (*op_mul)(Ta, Tx, Ty*, void*), void (*op_add)(Ty, Ty, Ty*, void*), void* vsp) {

  #pragma omp parallel for schedule(dynamic, 1)
  for(int p = 0 ; p < num_partitions ; p++)
  {
    for(int _row = partition_ptrs[p] ; _row < partition_ptrs[p+1] ; _row++)
    {
      int row = row_ids[_row];
      bool row_exists = get_bitvector(row, ybit_vector);
      Ty yval;
      if(row_exists)
      {
        yval = yvalue[row];
      }
      for (int nz = ia[_row]; nz < ia[_row + 1]; nz++) {
        Ty tmp_mul;
        int col_id = ja[nz];
        if(get_bitvector(col_id, xbit_vector))
        {
          op_mul(a[nz], xvalue[col_id], &tmp_mul, vsp);
          if(row_exists)
          {
            Ty tmp_add = yval;
            op_add(tmp_add, tmp_mul, &yval, vsp);
          } 
          else
          {
            yval = tmp_mul;
            row_exists=true;
          }
        }
      }
      if(row_exists)
      {
        set_bitvector(row, ybit_vector);
        yvalue[row] = yval;
      }
    }
  }
}

template <typename Ta, typename Tx, typename Ty>
void my_coospmspv(Ta* a, int* ia, int* ja, int num_partitions, int * partition_starts,
                  Tx* xvalue, int * xbit_vector,
                  Ty* yvalue, int * ybit_vector, int m, int n, int* nnz,
                  void (*op_mul)(Ta, Tx, Ty*, void*), void (*op_add)(Ty, Ty, Ty*, void*), void* vsp) {

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
        Ty tmp_mul;
        op_mul(a[nz], xvalue[col], &tmp_mul, vsp);
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



#endif  // SRC_SINGLENODE_SPMSPV_H_
