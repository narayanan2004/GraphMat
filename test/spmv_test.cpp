/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* ******************************************************************************/
/* Narayanan Sundaram (Intel Corp.), Michael Anderson (Intel Corp.)
 * ******************************************************************************/

#include <iostream>
#include "catch.hpp"
#include "generator.hpp"
#include <algorithm>
#include "test_utils.hpp"

template <typename TILE_T, typename EDGE_T>
void create_spmv_identity_test(int N) {
  // create identity matrix
  auto E1 = generate_identity_edgelist<EDGE_T>(N);
  GraphPad::SpMat<TILE_T> A;
  GraphPad::AssignSpMat(E1, &A, GraphPad::get_global_nrank(), GraphPad::get_global_nrank(), GraphPad::partition_fn_1d);

  //create random sparse vector
  auto E2 = generate_random_vector_edgelist<EDGE_T>(N, N/10);
  GraphPad::SpVec<GraphPad::DenseSegment<EDGE_T> > x;
  x.AllocatePartitioned(N, GraphPad::get_global_nrank(), GraphPad::vector_partition_fn);
  x.ingestEdgelist(E2);

  //create output vector
  GraphPad::SpVec<GraphPad::DenseSegment<EDGE_T> > y;
  y.AllocatePartitioned(N, GraphPad::get_global_nrank(), GraphPad::vector_partition_fn);
  GraphPad::Clear(&y);

  //do SPMV: y = A * x
  GraphPad::SpMSpV(A, x, &y, mul, add, NULL);

  //Collect elements of y
  GraphPad::edgelist_t<EDGE_T> E3;
  GraphPad::edgelist_t<EDGE_T> E4;
  y.get_edges(&E3);
  collect_edges(E3, E4);
  std::sort(E4.edges, E4.edges + E4.nnz, edge_compare<EDGE_T>);

  //Collect elements of x
  GraphPad::edgelist_t<EDGE_T> E5;
  collect_edges(E2, E5);
  std::sort(E5.edges, E5.edges + E5.nnz, edge_compare<EDGE_T>);

  //Compare x == y
  REQUIRE(x.getNNZ() == y.getNNZ());
  REQUIRE(E4.nnz == E5.nnz);
  for (int i = 0; i < E4.nnz; i++) {
    REQUIRE(E5.edges[i].dst == E4.edges[i].dst);
    REQUIRE(E5.edges[i].src == E4.edges[i].src);
    REQUIRE(E5.edges[i].val == E4.edges[i].val);
  }
}


TEST_CASE("spmv", "spmv") 
{
  SECTION("CSR mvm") {
    create_spmv_identity_test<GraphPad::CSRTile<double>, double>(5000);
    create_spmv_identity_test<GraphPad::CSRTile<double>, double>(10);
  }
  SECTION("DCSC mvm") {
    create_spmv_identity_test<GraphPad::DCSCTile<double>, double>(5000);
    create_spmv_identity_test<GraphPad::DCSCTile<double>, double>(10);
  }
  SECTION("COO mvm") {
    create_spmv_identity_test<GraphPad::COOTile<double>, double>(5000);
    create_spmv_identity_test<GraphPad::COOTile<double>, double>(10);
  }
  SECTION("COOSIMD32 mvm") {
    create_spmv_identity_test<GraphPad::COOSIMD32Tile<double>, double>(5000);
    create_spmv_identity_test<GraphPad::COOSIMD32Tile<double>, double>(10);
  }

}
