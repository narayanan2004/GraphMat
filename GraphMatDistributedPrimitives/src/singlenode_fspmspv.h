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


#ifndef SRC_SINGLENODE_FSPMSPV_H_
#define SRC_SINGLENODE_FSPMSPV_H_
#include <xmmintrin.h>
#include "src/bitvector.h"


template <typename Ta, typename Tx, typename Ty>
void my_csrfspmspv(Ta* a, int* ia, int* ja, Tx* xvalue, int * xbit_vector,
                  int * mbit_vector,
                  Ty* yvalue, int * ybit_vector, int m, int n,
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
    for(int start_row32 = start_row ; start_row32 < end_row ; start_row32+=32)
    {
      if(_popcnt32(mbit_vector[start_row32 / 32]) == 0) continue;
      for(int row = start_row32 ; row < start_row32+32; row++)
      {
        if(!get_bitvector(row, mbit_vector)) continue;
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
  }
  //*nnz = m * n;
}

#endif  // SRC_SINGLENODE_FSPMSPV_H_
