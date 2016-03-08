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


#ifndef SRC_MULTINODE_SPARSIFY_H_
#define SRC_MULTINODE_SPARSIFY_H_

#include "src/SpMat.h"
#include "src/SpVec.h"
#include "src/TileOps.h"

template <template <typename> class SpSegmenta, template <typename> class SpSegmentb, typename Ta, typename Tb>
void Sparsify_tile(const SpVec<SpSegmenta<Ta> > & v_in, SpVec<SpSegmentb<Tb> > * v_out, int start, int end,
                 void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {
  for (int i = start; i < end; i++) {
    if (v_in.nodeIds[i] == global_myrank) {
      sparsify_segment(v_in.segments[i], &(v_out->segments[i]), add_fp, vsp);
    }
  }
}

template <template <typename> class SpTilea, template <typename> class SpTileb, typename Ta, typename Tb>
void Sparsify_tile(const SpMat<SpTilea<Ta> > & v_in, SpMat<SpTileb<Tb> > * v_out, int start_x, int start_y, int end_x, int end_y,
                 void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp) {
  for (int i = start_y; i < end_y; i++) {
    for(int j = start_x ; j < end_x ; j++)
    {
      if (v_in.nodeIds[i + j * v_in.ntiles_y] == global_myrank) {
        sparsify_tile(v_in.tiles[i][j], &(v_out->tiles[i][j]), add_fp, vsp);
      }
    }
  }
}


#endif  // SRC_MULTINODE_SPARSIFY_H_
