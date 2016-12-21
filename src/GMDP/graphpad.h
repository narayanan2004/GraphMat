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


#ifndef SRC_GRAPH_BLAS_H_
#define SRC_GRAPH_BLAS_H_

#include <mpi.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <parallel/algorithm>

#include <map>
#include <set>
#include <string>
#include <list>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <climits>

#include <sstream>
#include <fstream>
#include <iostream>

namespace GraphPad {

#include "GMDP/matrices/edgelist.h"
#include "GMDP/matrices/SpMat.h"
#include "GMDP/matrices/layouts.h"
#include "GMDP/matrices/COOSIMD32Tile.h"
#include "GMDP/matrices/COOTile.h"
#include "GMDP/matrices/CSRTile.h"
#include "GMDP/matrices/DCSCTile.h"
#include "GMDP/matrices/HybridTile.h"
#include "GMDP/matrices/SpMat.h"
#include "GMDP/vectors/SpVec.h"
#include "GMDP/vectors/DenseSegment.h"
#include "GMDP/multinode/intersectreduce.h"
#include "GMDP/multinode/unionreduce.h"
#include "GMDP/multinode/reduce.h"
#include "GMDP/multinode/spmspv.h"
#include "GMDP/multinode/spmspv3.h"
#include "GMDP/multinode/apply.h"
#include "GMDP/multinode/clear.h"

inline double get_compression_threshold() {
  return 0.5;
}

template <typename T>
void ReadEdgesBin(edgelist_t<T>* edgelist, const char* fname_in, bool randomize) {
  load_edgelist(fname_in, get_global_myrank(), get_global_nrank(), edgelist);

  if (randomize) {
    randomize_edgelist_square<T>(edgelist, get_global_nrank());
  }
}

template <typename T>
void ReadEdgesTxt(edgelist_t<T>* edgelist, const char* fname_in, bool randomize) {
  load_edgelist_txt(fname_in, get_global_myrank(), get_global_nrank(), edgelist);

  if (randomize) {
    randomize_edgelist_square<T>(edgelist, get_global_nrank());
  }
}

template <typename T>
void WriteEdgesTxt(const edgelist_t<T>& edgelist, const char* fname_in) {
  write_edgelist_txt(fname_in, get_global_myrank(), get_global_nrank(), edgelist);
}

template <typename T>
void WriteEdgesBin(const edgelist_t<T>& edgelist, const char* fname_in) {
  write_edgelist_bin(fname_in, get_global_myrank(), get_global_nrank(), edgelist);
}


template <template <typename> class SpTile, typename T>
void AssignSpMat(edgelist_t<T> edgelist, SpMat<SpTile<T> >* mat, int ntx,
                 int nty, int (*pfn)(int, int, int, int, int)) {
  mat->Allocate2DPartitioned(edgelist.m, edgelist.n, ntx, nty, pfn);
  mat->ingestEdgelist(edgelist);
  mat->print_tiles("A", 0);
}

template <typename T>
void AssignSpVec(edgelist_t<T> edgelist, SpVec<T>* vec, int ntx,
                 int (*pfn)(int, int, int)) {
  vec->AllocatePartitioned(edgelist.m, ntx, pfn);
  vec->ingestEdgelist(edgelist);
}

template <template <typename> class SpTile, typename T>
void WriteTxt(SpMat<SpTile<T> >* mat, char* fname_in) {
  // Print element 0 for now
  int global_myrank = get_global_myrank();
  if (global_myrank == 0) {
    std::cout << "Element 0: " << mat->tiles[0][0].a[0] << std::endl;
  }
}

template <typename T>
void WriteTxt(SpVec<T>* vec, char* fname_in) {
  // Print element 0 for now
  int global_myrank = get_global_myrank();
  if (global_myrank == 0) {
    std::cout << "Element 0: " << vec->segments[0]->value[0] << std::endl;
  }
}

template <template <typename> class SpTile, typename T>
void Transpose(const SpMat<SpTile<T> >& mat, SpMat<SpTile<T> >* matc, int ntx,
               int nty, int (*pfn)(int, int, int, int, int)) {

  edgelist_t<T> edgelist;
  mat.get_edges(&edgelist);
#pragma omp parallel for
  for (int i = 0; i < edgelist.nnz; i++) {
    int tmp = edgelist.edges[i].src;
    edgelist.edges[i].src = edgelist.edges[i].dst;
    edgelist.edges[i].dst = tmp;
  }
  int tmp = edgelist.m;
  edgelist.m = edgelist.n;
  edgelist.n = tmp;
  SpMat<SpTile<T> > C;

  AssignSpMat(edgelist, &C, ntx, nty, pfn);
  if(edgelist.nnz > 0)
  {
    _mm_free(edgelist.edges);
  }
  *matc = C;
}

template <template <typename> class SpTile, template <typename> class SpSegment, typename Ta, typename Tx,
          typename Ty>
void SpMSpV(const SpMat<SpTile<Ta> >& mata, const SpVec<SpSegment<Tx> >& vecx,
            SpVec<SpSegment<Ty> >* vecy, void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp=NULL) {
  SpMSpV_tile(mata, vecx, vecy, 0, 0, mata.ntiles_y, mata.ntiles_x, mul_fp,
              add_fp, vsp);
}

template <template <typename> class SpTile, template <typename> class SpSegment, typename Ta, typename Tx,
          typename Tvp, typename Ty>
void SpMSpV3(const SpMat<SpTile<Ta> >& mata, const SpVec<SpSegment<Tx> >& vecx,
            const SpVec<SpSegment<Tvp> > & vecvp, 
	    SpVec<SpSegment<Ty> >* vecy, void (*mul_fp)(Ta, Tx, Tvp, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp=NULL) {
  SpMSpV3_tile(mata, vecx, vecvp, vecy, 0, 0, mata.ntiles_y, mata.ntiles_x, mul_fp,
              add_fp, vsp);
}

template <template<typename> class SpSegment, typename T, typename VT>
void MapReduce(SpVec<SpSegment<VT> > * vec, T* res, void (*op_map)(VT*, T*, void*), void (*op_fp)(T, T, T*, void*), void* vsp=NULL) {
  MapReduce_tile(vec, res, 0, vec->nsegments, op_map, op_fp, vsp);
}

template <template<typename> class SpSegment, typename Ta, typename Tb>
void Apply(const SpVec<SpSegment<Ta> > & v_in, SpVec<SpSegment<Tb> > * v_out, void (*add_fp)(Ta, Tb*, void*), void* vsp=NULL)
{
  Apply_tile(v_in, v_out, 0, v_in.nsegments, add_fp, vsp);
}

template <typename T>
void Clear(SpVec<T> * v1)
{
  Clear_tile(v1, 0, v1->nsegments);
}

template <template <typename> class SpSegment, typename Ta, typename Tb, typename Tc>
void IntersectReduce(const SpVec<SpSegment<Ta> > & v1, const SpVec<SpSegment<Tb> > & v2, SpVec<SpSegment<Tc> > * v3, void (*op_fp)(Ta,Tb,Tc*,void*), void* vsp=NULL)
{
  IntersectReduce_tile(v1, v2, v3, 0, v1.nsegments, op_fp, vsp);
}

template <typename Ta, typename Tb, typename Tc>
void UnionReduce(const SpVec<Ta> & v1, const SpVec<Tb> & v2, SpVec<Tc> * v3, void (*op_fp)(Ta,Tb,Tc*,void*), void* vsp=NULL)
{
  UnionReduce_tile(v1, v2, v3, 0, v1.nsegments, op_fp, vsp);
}

}  // namespace GraphPad

#endif  // SRC_GRAPH_BLAS_H_
