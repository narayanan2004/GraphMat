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


#ifndef SRC_SINGLENODE_INTERSECTREDUCE_H_
#define SRC_SINGLENODE_INTERSECTREDUCE_H_

#include <algorithm>
#include "GMDP/utils/bitvector.h"

template <typename Ta, typename Tb, typename Tc>
void intersect_dense_segment(Ta* v1, int * bv1, int * nnz, int num_ints, Tb * v2, int * bv2, Tc * v3, int * bv3,
                          void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {

  #pragma omp parallel for
  for(int i = 0 ; i < num_ints ; i++)
  {
    bv3[i] = bv1[i] & bv2[i];
  }

  int tmp_nnz = 0;
  #pragma omp parallel for reduction(+:tmp_nnz)
  for(int ii = 0 ; ii < num_ints ; ii++)
  {
    int cnt = _popcnt32(bv3[ii]);
    if(cnt == 0) continue;
    tmp_nnz += cnt;
    for(int i = ii*32 ; i < (ii+1)*32 ; i++)
    {
      if(get_bitvector(i, bv3))
      {
        Ta tmp = v1[i];
	op_fp(v1[i], v2[i], &(v3[i]), vsp);
      }
    }
  }
  *nnz = tmp_nnz;
}

template <typename Ta, typename Tb, typename Tc>
void intersect_segment(const DenseSegment<Ta> * s1, const DenseSegment<Tb> * s2, DenseSegment<Tc> * s3, 
                    void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {

  s3->alloc();
  s3->initialize();
  if(!s1->properties->uninitialized && !s2->properties->uninitialized)
  {
    intersect_dense_segment(s1->properties->value, s1->properties->bit_vector, &(s3->properties->nnz), s1->num_ints, s2->properties->value, s2->properties->bit_vector, s3->properties->value, s3->properties->bit_vector, op_fp, vsp);
  }
}

#endif  // SRC_SINGLENODE_INTERSECTREDUCE_H_
