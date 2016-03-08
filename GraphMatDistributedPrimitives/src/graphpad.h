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
#include <mkl.h>
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

uint64_t mul_flops = 0;
uint64_t add_flops = 0;
uint64_t add_flops2 = 0;
uint64_t nnz_C = 0;

double spmspv_send_time = 0;
double spmspv_mult_time = 0;
double spmspv_reduce_send_time = 0;
double spmspv_reduce_time = 0;

int global_myrank = 0;
int global_nrank = 0;

double COMPRESSION_THRESHOLD;

#include "src/layouts.h"
#include "src/edgelist.h"
#include "src/SpMat.h"
#include "src/SpVec.h"
#include "src/multinode_fspgemm.h"
#include "src/multinode_fspmspv.h"
#include "src/multinode_spgemm.h"
#include "src/multinode_intersectreduce.h"
#include "src/multinode_unionreduce.h"
#include "src/multinode_reduce.h"
#include "src/multinode_spmspv.h"
#include "src/multinode_spmspv3.h"
#include "src/multinode_apply.h"
#include "src/multinode_sparsify.h"
#include "src/multinode_clear.h"

template <typename T>
void ReadEdgesBin(edgelist_t<T>* edgelist, const char* fname_in, bool randomize) {
  load_edgelist(fname_in, global_myrank, global_nrank, edgelist);

  if (randomize) {
    randomize_edgelist_square<T>(edgelist, global_nrank);
  }
}

template <typename T>
void ReadEdgesTxt(edgelist_t<T>* edgelist, const char* fname_in, bool randomize) {
  load_edgelist_txt(fname_in, global_myrank, global_nrank, edgelist);

  if (randomize) {
    randomize_edgelist_square<T>(edgelist, global_nrank);
  }
}

template <typename T>
void WriteEdgesTxt(const edgelist_t<T>& edgelist, const char* fname_in) {
  write_edgelist_txt(fname_in, global_myrank, global_nrank, edgelist);
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
  if (global_myrank == 0) {
    std::cout << "Element 0: " << mat->tiles[0][0].a[0] << std::endl;
  }
}

template <typename T>
void WriteTxt(SpVec<T>* vec, char* fname_in) {
  // Print element 0 for now
  if (global_myrank == 0) {
    std::cout << "Element 0: " << vec->segments[0]->value[0] << std::endl;
  }
}

template <template <typename> class ATile, template <typename> class BTile,
          template <typename> class CTile, typename Ta, typename Tb,
          typename Tc>
void SpGEMM(const SpMat<ATile<Ta> >& mata, const SpMat<BTile<Tb> >& matb,
            SpMat<CTile<Tc> >* matc, void (*mul_fp)(Ta, Tb, Tc*, void*),
            void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp=NULL) {
  if(matc->empty)
  {
    matc->Allocate2DPartitioned(mata.m, matb.n, mata.num_tiles_x, mata.num_tiles_y,
                               mata.pfn);
  }
  SpGEMM_tile_outerproduct(mata, matb, matc, 0, 0, 0, mata.ntiles_y,
                           matb.ntiles_x, mata.ntiles_x, mul_fp, add_fp, vsp);
  // SpGEMM_tile_innerproduct(mata, matb, &Cout, 0, 0, 0, mata.ntiles_y,
  //                         matb.ntiles_x, mata.ntiles_x, mul_fp, add_fp);
}

template <template <typename> class SpTile, typename Ta, typename Tb,
          typename Tc, typename Tf>
SpMat<SpTile<Tc> >* FSpGEMM(const SpMat<SpTile<Ta> >& mata,
                            const SpMat<SpTile<Tb> >& matb,
                            SpMat<SpTile<Tc> >* matc,
                            const SpMat<SpTile<Tf> >& matf,
                            void (*mul_fp)(Ta, Tb, Tc*, void*), void (*add_fp)(Tc, Tc, Tc*, void*), void* vsp=NULL) {
  // Create empty C with correct partitioning (If it is not there and correctly
  // partitioned already)
  SpMat<SpTile<Tc> > C;
  C.Allocate2DPartitioned(mata.m, matb.n, mata.num_tiles_x, mata.num_tiles_y,
                          mata.pfn);

  // SpGEMM_tile accumulates into C using add_fp
  FSpGEMM_tile(mata, matb, &C, matf, 0, 0, 0, mata.ntiles_y, matb.ntiles_x,
               mata.ntiles_x, mul_fp, add_fp, vsp);
  // SpGEMM_tile_innerproduct<Ta, Tb, Tc> (mata, matb, C, 0, 0, 0,
  // mata.ntiles_y, matb.ntiles_x, mata.ntiles_x, mul_fp, add_fp);

  //  If we created an empty C, then we must copy it over to matc
  *matc = C;
}

template <template <typename> class SpTilea, template <typename> class SpTileb, typename Ta, typename Tb>
void Sparsify(const SpMat<SpTilea<Ta> > & v_in, SpMat<SpTileb<Tb> > * v_out, void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp=NULL)
{
  printf("Sparsify graph_blas.h\n");
  Sparsify_tile(v_in, v_out, 0, 0, v_in.ntiles_x, v_in.ntiles_y, add_fp, vsp);
}

template <template <typename> class SpTilea, template <typename> class SpTileb, typename Ta, typename Tb>
void Apply(const SpMat<SpTilea<Ta> > & v_in, SpMat<SpTileb<Tb> > * v_out, void (*add_fp)(Ta, Tb*, void*), void* vsp=NULL)
{
  Apply_tile(v_in, v_out, 0, 0, v_in.ntiles_x, v_in.ntiles_y, add_fp, vsp);
}

template <template <typename> class SpTile, typename T>
void Clear(SpMat<SpTile <T> > * v1)
{
  Clear_tile(v1, 0, 0, v1->ntiles_x, v1->ntiles_y);
}

template <template <typename> class SpTile, typename Ta, typename Tb,
          typename Tc>
void IntersectReduce(const SpMat<SpTile<Ta> >& mata,
                     const SpMat<SpTile<Tb> >& matb, SpMat<SpTile<Tc> >* matc,
                     void (*op_fp)(Ta, Tb, Tc*, void*), void* vsp=NULL) {
  SpMat<SpTile<Tc> > C;
  C.Allocate2DPartitioned(mata.m, mata.n, mata.num_tiles_x, mata.num_tiles_y,
                          mata.pfn);

  assert(mata.m == matb.m);
  assert(mata.m == C.m);
  assert(mata.n == matb.n);
  assert(mata.n == C.n);

  double start_time = MPI_Wtime();
  IntersectReduce_tile(mata, matb, &C, 0, 0, mata.ntiles_x, mata.ntiles_y,
                       op_fp, vsp);
  double end_time = MPI_Wtime();
  std::cout << "Intersect time: " << end_time - start_time << std::endl;

  assert(mata.m == C.m);
  assert(mata.n == C.n);
  *matc = C;
}

template <template <typename> class SpTile, typename T>
SpMat<SpTile<T> >* UnionReduce(const SpMat<SpTile<T> >& mata,
                               SpMat<SpTile<T> >* matb, T (*op_fp)(T, T)) {
  double start_time = MPI_Wtime();
  UnionReduce_tile(mata, matb, 0, 0, mata.ntiles_x, mata.ntiles_y, op_fp);
  double end_time = MPI_Wtime();
  std::cout << "Union time: " << end_time - start_time << std::endl;
}

template <template <typename> class SpTile, typename T>
void Reduce(const SpMat<SpTile<T> >& mata, T* res, void (*op_fp)(T, T, T*, void*), void* vsp=NULL) {
  Reduce_tile(mata, res, 0, 0, mata.ntiles_x, mata.ntiles_y, op_fp, vsp);
}

template <template <typename> class SpTile, template <typename> class SpSegment, typename T>
void Reduce(const SpMat<SpTile<T> >& mata, SpVec<SpSegment<T> > * res, void (*op_fp)(T, T, T*, void*), void* vsp=NULL) {
  Reduce_tile(mata, res, 0, 0, mata.ntiles_x, mata.ntiles_y, op_fp, vsp);
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
          typename Ty, typename Tm>
void FSpMSpV(const SpMat<SpTile<Ta> >& mata, const SpVec<SpSegment<Tx> >& vecx, const SpVec<SpSegment<Tm> >& vecm,
            SpVec<SpSegment<Ty> >* vecy, void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp=NULL) {
  FSpMSpV_tile(mata, vecx, vecm, vecy, 0, 0, mata.ntiles_y, mata.ntiles_x, mul_fp,
              add_fp, vsp);
}


template <template <typename> class SpTile, template <typename> class SpSegment, typename Ta, typename Tx,
          typename Ty>
void SpMSpV(const SpMat<SpTile<Ta> >& mata, const SpVec<SpSegment<Tx> >& vecx,
            SpVec<SpSegment<Ty> >* vecy, void (*mul_fp)(Ta, Tx, Ty*, void*), void (*add_fp)(Ty, Ty, Ty*, void*), void* vsp=NULL) {
  SpMSpV_tile(mata, vecx, vecy, 0, 0, mata.ntiles_y, mata.ntiles_x, mul_fp,
              add_fp, vsp);
}

template <template <typename> class SpTile, typename Ta, typename Tx,
          typename Tvp, typename Ty>
void SpMSpV3(const SpMat<SpTile<Ta> >& mata, const SpVec<Tx>& vecx,
            const SpVec<Tvp> & vecvp, 
	    SpVec<Ty>* vecy, Ty (*mul_fp)(Ta, Tx, Tvp), Ty (*add_fp)(Ty, Ty)) {
  SpMSpV3_tile(mata, vecx, vecvp, vecy, 0, 0, mata.ntiles_y, mata.ntiles_x, mul_fp,
              add_fp);
}

template <template<typename> class SpSegment, typename T>
void Reduce(const SpVec<SpSegment<T> >& vec, T* res, void (*op_fp)(T, T, T*, void*), void* vsp=NULL) {
  Reduce_tile(vec, res, 0, vec.nsegments, op_fp, vsp);
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

template <template <typename> class SpSegmenta, template <typename> class SpSegmentb, typename Ta, typename Tb>
void Sparsify(const SpVec<SpSegmenta<Ta> > & v_in, SpVec<SpSegmentb<Tb> > * v_out, void (*add_fp)(Ta, bool*, Tb*, void*), void* vsp=NULL)
{
  Sparsify_tile(v_in, v_out, 0, v_in.nsegments, add_fp, vsp);
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

void GB_Init() {
  MPI_Comm_rank(MPI_COMM_WORLD, &global_myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_nrank);
  COMPRESSION_THRESHOLD = 0.5;
}
}  // namespace GraphPad

#endif  // SRC_GRAPH_BLAS_H_
