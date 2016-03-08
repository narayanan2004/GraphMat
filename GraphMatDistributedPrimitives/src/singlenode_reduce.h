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


#ifndef SRC_SINGLENODE_REDUCE_H_
#define SRC_SINGLENODE_REDUCE_H_

template <typename T>
void reduce_csr(T* a, int nnz, T* result, bool* res_set, void (*op_fp)(T, T, T*, void*), void* vsp) {
  if (nnz == 0) return;
  if (!(*res_set)) {
    (*result) = a[0];
    for (int ii = 1; ii < nnz; ii++) {
      // std::cout << ii << "\t" << a[ii] << std::endl;
      T res_tmp = *result;
      op_fp(res_tmp, a[ii], result, vsp);
      (*res_set) = true;
    }
  } else {
    for (int ii = 0; ii < nnz; ii++) {
      T res_tmp = *result;
      op_fp(res_tmp, a[ii], result, vsp);
    }
  }
}

template <typename T>
void reduce_dcsc(T* vals, int nnz, T* result, bool* res_set, void (*op_fp)(T, T, T*, void*), void* vsp) {
  if (nnz == 0) return;
  if (!(*res_set)) {
    (*result) = vals[0];
    for (int ii = 1; ii < nnz; ii++) {
      T res_tmp = *result;
      op_fp(res_tmp, vals[ii], result, vsp);
      (*res_set) = true;
    }
  } else {
    for (int ii = 0; ii < nnz; ii++) {
      T res_tmp = *result;
      op_fp(res_tmp, vals[ii], result, vsp);
    }
  }
}


template <typename T>
void reduce_dcsc(int* row_inds, int* col_ptrs, int* col_indices, T* vals,
               int num_partitions, int* row_pointers, int* col_starts,
               int* edge_pointers, T* yvalue,
               int * ybit_vector, void (*op_fp)(T, T, T*, void*), void* vsp) {

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
      {
        int nz_idx = col_ptrs_cur[j];
        for (; nz_idx < col_ptrs_cur[j + 1]; nz_idx++) {
          int row_ind = partitioned_row_offset[nz_idx];
          Ta Aval = partitioned_val_offset[nz_idx];
	  Ty temp_mul_result;
          if(get_bitvector(row_ind, ybit_vector))
	  {
	    Ty temp_y_copy = yvalue[row_ind];
            op_add(Aval, temp_y_copy, &(yvalue[row_ind]));
	  }
	  else
	  {
            yvalue[row_ind] = Aval;
            set_bitvector(row_ind, ybit_vector);
	  }
        }
      }
    }
  }
}


template <typename T>
void reduce_dense_segment(T* value, int * bitvector, int nnz, T* result, bool* res_set,
                          void (*op_fp)(T, T, T*, void*), void* vsp) {

  for(int i = 0 ; i < nnz ; i++)
  {
    if(get_bitvector(i, bitvector))
    {
      T temp_result = *result;
	  op_fp(temp_result, value[i], result, vsp);
    }
  }
}


template <typename VT, typename T>
void mapreduce_dense_segment(VT* value, int * bitvector, int nnz, T* result, bool* res_set,
                          void (*op_map)(VT*, T*, void*), void (*op_fp)(T, T, T*, void*), void* vsp) {

  for(int i = 0 ; i < nnz ; i++)
  {
    if(get_bitvector(i, bitvector))
    {
      T temp_result = *result;
      T temp_result2;
      op_map(value + i, &temp_result2, vsp);
	  op_fp(temp_result, temp_result2, result, vsp);
    }
  }
}


template <typename T>
void reduce_dense(T* value, bool* bitvector, int m, int n, T* result,
                  bool* res_set, T (*op_fp)(T, T)) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (bitvector[i + j * m]) {
        if (!(*res_set)) {
          (*res_set) = true;
          (*result) = value[i + j * m];
        } else {
	  T tmp = value[i + j * m];
          (*result) = op_fp(tmp, (*result));
        }
      }
    }
  }
}

template <typename T>
void reduce_dense(T* value, int * bitvector, int m, int n, T* result, int * result_bitvector,
                  void (*op_fp)(T, T, T*, void*), void* vsp) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if(get_bitvector(i + j * m, bitvector))
      {
        if(!get_bitvector(i, result_bitvector))
	{
	  set_bitvector(i, result_bitvector);
	  result[i] = value[i + j * m];
        } else {
	  T tmp_result = result[i];
	  op_fp(value[i + j * m], tmp_result, &(result[i]), vsp);
        }
      }
    }
  }
}



#endif  // SRC_SINGLENODE_REDUCE_H_
