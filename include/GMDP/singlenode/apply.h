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


#ifndef SRC_SINGLENODE_APPLY_H_
#define SRC_SINGLENODE_APPLY_H_

template <typename Ta, typename Tb>
void apply_dense_segment(Ta* v1, int * bitvector, int * nnz, int num_ints,
                         Tb* v2, int * bitvector2, 
                          void (*add_fp)(Ta, Tb*, void*), void* vsp) {
  #pragma omp parallel for
  for(int i = 0 ; i < num_ints ; i++)
  {
    bitvector2[i] = bitvector2[i] | bitvector[i];
  }

  int tmp_nnz = 0;
  #pragma omp parallel for reduction(+:tmp_nnz)
  for(int ii = 0 ; ii < num_ints ; ii++)
  {
    int cnt = _popcnt32(bitvector[ii]);
    if(cnt == 0) continue;
    //if(_popcnt32(bitvector[ii]) == 0) continue;
    tmp_nnz += cnt;
    for(int i = ii*32 ; i < (ii+1)*32 ; i++)
    {
      if(get_bitvector(i, bitvector))
      {
        Ta tmp = v1[i];
        add_fp(tmp, &(v2[i]), vsp);
      }
    }
  }
 *nnz = tmp_nnz;
}

template <typename Ta, typename Tb>
void apply_segment(const DenseSegment<Ta> * s_in, DenseSegment<Tb> * s_out, 
                    void (*add_fp)(Ta, Tb*, void*), void* vsp) {
  s_out->alloc();
  s_out->initialize();
  apply_dense_segment(s_in->properties->value, s_in->properties->bit_vector, &(s_out->properties->nnz), s_in->num_ints, s_out->properties->value, s_out->properties->bit_vector,  add_fp, vsp);
}



#endif  // SRC_SINGLENODE_APPLY_H_
