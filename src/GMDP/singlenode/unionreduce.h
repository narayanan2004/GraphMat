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


#ifndef SRC_SINGLENODE_UNIONREDUCE_H_
#define SRC_SINGLENODE_UNIONREDUCE_H_

#include <algorithm>

template <typename Ta, typename Tb, typename Tc>
void union_dense(Ta* v1, int * bv1, int nnz, int num_ints, Tb * v2, int * bv2, Tc * v3, int * bv3,
                          void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) 
{

  #pragma omp parallel for
  for (int ii = 0; ii < nnz; ii++) {
    bool set1 = get_bitvector(ii, bv1);
    bool set2 = get_bitvector(ii, bv2);
    if(set1 && !set2)
    {
      v3[ii] = v1[ii];
    }
    else if(!set1 && set2)
    {
      v3[ii] = v2[ii];
    }
    else if(set1 && set2)
    {
      op_fp(v1[ii], v2[ii], &(v3[ii]), vsp);
    }
  }

  #pragma omp parallel for
  for(int i = 0 ; i < num_ints ; i++)
  {
    bv3[i] = bv1[i] | bv2[i];
  }
}

template <typename Ta, typename Tb>
void union_compressed(Ta* v1, int* indices, int nnz, int capacity, int num_ints, Tb * v2, int * bv2,
                          void (*op_fp)(Ta, Tb, Tb*, void*), void* vsp) 
{
  //int * indices = reinterpret_cast<int*>(v1 + nnz);
  int npartitions = omp_get_max_threads() * 16;
  #pragma omp parallel for
  for(int p = 0 ; p < npartitions ; p++)
  {
    int nz_per = (nnz + npartitions - 1) / npartitions;
    int start_nnz = p * nz_per;
    int end_nnz = (p+1) * nz_per;
    if(end_nnz > nnz) end_nnz = nnz;

    // Adjust
    if(start_nnz > 0)
    {
      while((start_nnz < nnz) && (indices[start_nnz]/32 == indices[start_nnz-1]/32)) start_nnz++;
    }
    while((end_nnz < nnz) && (indices[end_nnz]/32 == indices[end_nnz-1]/32)) end_nnz++;

    for(int i = start_nnz  ; i < end_nnz ; i++)
    {
      int idx = indices[i];
      if(get_bitvector(idx, bv2))
      {
        //Tb tmp = v2[idx];
        //op_fp(v1[i], tmp, &(v2[idx]), vsp);
        op_fp(v1[i], v2[idx], &(v2[idx]), vsp);
      }
      else
      {
        set_bitvector(idx, bv2);
        v2[idx] = v1[i];
      }
    }
  }
}

#endif  // SRC_SINGLENODE_UNIONREDUCE_H_
