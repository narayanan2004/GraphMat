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


#ifndef SRC_MULTINODE_INTERSECTREDUCE_H_
#define SRC_MULTINODE_INTERSECTREDUCE_H_

#include "GMDP/vectors/SpVec.h"
#include "GMDP/singlenode/intersectreduce.h"

template <template <typename> class SpSegment, typename Ta, typename Tb, typename Tc>
void IntersectReduce(const SpVec<SpSegment<Ta> > * veca,
                          const SpVec<SpSegment<Tb> > * vecb,
                          SpVec<SpSegment<Tc> > * vecc,
                          void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp) {
  int global_myrank = get_global_myrank();
  int start = 0;
  int end = veca->nsegments;
  for (int i = start; i < end; i++) {
    if (veca->nodeIds[i] == global_myrank) {
      intersect_segment(veca->segments[i], vecb->segments[i], vecc->segments[i], op_fp, vsp);
    }
  }
}

#endif  // SRC_MULTINODE_INTERSECTREDUCE_H_
