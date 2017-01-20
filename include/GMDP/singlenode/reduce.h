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
void reduce_dense_segment(T* value, int * bitvector, int nnz, T* result, bool* res_set,
                          void (*op_fp)(T, T, T*, void*), void* vsp) {

  for(int i = 0 ; i < nnz ; i++)
  {
    if(get_bitvector(i, bitvector))
    {
      //T temp_result = *result;
      op_fp(*result, value[i], result, vsp);
    }
  }
}


template <typename VT, typename T>
void mapreduce_dense_segment(VT* value, int * bitvector, int nnz, T* result, bool* res_set,
                          void (*op_map)(VT*, T*, void*), void (*op_fp)(T, T, T*, void*), void* vsp) {

  int nthreads = omp_get_max_threads();
  T * local_reduced = new T[nthreads*16];
  bool * firstSet = new bool[nthreads*16];
  #pragma omp parallel for
  for(int p = 0 ; p < nthreads ; p++)
  {
    firstSet[p*16] = false;
    int nnz_per_thread = (nnz + nthreads - 1) / nthreads;
    int start = nnz_per_thread * p;
    int end = nnz_per_thread * (p+1);
    if(start > nnz) start = nnz;
    if(end > nnz) end  = nnz;

    for(int i = start ; i < end ; i++)
    {
      if(get_bitvector(i, bitvector))
      {
        T temp_result2;
        op_map(value + i, &temp_result2, vsp);
        if(firstSet[p*16])
        {
          T temp_result = local_reduced[p*16];
  	  op_fp(temp_result, temp_result2, local_reduced + p*16, vsp);
        } 
        else
        {
          local_reduced[p*16] = temp_result2;
          firstSet[p*16] = true;
        }
      }
    }
  }

  // Reduce each thread's local result
  for(int p = 0 ; p < nthreads ; p++)
  {
    if(firstSet[p*16])
    {
      //T temp_result = *result;
      op_fp(*result, local_reduced[p*16], result, vsp);
    }
  }
  delete [] local_reduced;
  delete [] firstSet;
}

template <typename T>
void reduce_segment(const DenseSegment<T> * segment, T* res, bool* res_set,
                    void (*op_fp)(T, T, T*, void*), void* vsp) {

  reduce_dense_segment(segment->properties->value, segment->properties->bit_vector, segment->capacity, res, res_set, op_fp, vsp);
}

template <typename VT, typename T>
void mapreduce_segment(DenseSegment<VT> * segment, T* res, bool* res_set,
                    void (*op_map)(VT*, T*, void*), void (*op_fp)(T, T, T*, void*), void* vsp) {
  segment->alloc();
  segment->initialize();
  mapreduce_dense_segment(segment->properties->value, segment->properties->bit_vector, segment->capacity, res, res_set, op_map, op_fp, vsp);
}

#endif  // SRC_SINGLENODE_REDUCE_H_
