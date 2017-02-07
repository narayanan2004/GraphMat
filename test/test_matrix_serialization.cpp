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
#include "generator.h"
#include <algorithm>
#include "test_utils.h"

template <typename TILE_T, typename EDGE_T>
void matrix_serialization_test(int N)
{
  auto E = generate_random_edgelist<EDGE_T>(N, 16);
  GraphMat::SpMat<TILE_T>* A;
  GraphMat::SpMat<TILE_T>* B;

  // Build A
  A = new GraphMat::SpMat<TILE_T>(E, GraphMat::get_global_nrank(), GraphMat::get_global_nrank(), GraphMat::partition_fn_1d);

  // Serialize A
  std::stringstream ss;
  {
    boost::archive::binary_oarchive bo(ss);
    bo << A;
  }

  // Deserialize A into B
  {
    boost::archive::binary_iarchive bi(ss);
    bi >> B;
  }

  GraphMat::edgelist_t<EDGE_T> OE1;
  GraphMat::edgelist_t<EDGE_T> OE2;
  A->get_edges(&OE1);
  B->get_edges(&OE2);

  // Validate A == B;
  std::sort(OE1.edges, OE1.edges + OE1.nnz, edge_compare<EDGE_T>);
  std::sort(OE2.edges, OE2.edges + OE2.nnz, edge_compare<EDGE_T>);

  REQUIRE(OE1.nnz == OE2.nnz);
  REQUIRE(OE1.m == OE2.m);
  REQUIRE(OE1.n == OE2.n);
  for(int i = 0 ; i < OE1.nnz ; i++)
  {
          REQUIRE(OE1.edges[i].src == OE2.edges[i].src);
          REQUIRE(OE1.edges[i].dst == OE2.edges[i].dst);
          REQUIRE(OE1.edges[i].val == OE2.edges[i].val);
  }
}


TEST_CASE("matrix_serialization", "matrix_serialization")
{
  SECTION(" DCSCTile basic tests ", "CSRTile basic tests") {
        matrix_serialization_test<GraphMat::DCSCTile<int>, int>(5);
        matrix_serialization_test<GraphMat::DCSCTile<int>, int>(500);
  }
  SECTION(" CSRTile basic tests ", "CSRTile basic tests") {
        matrix_serialization_test<GraphMat::CSRTile<int>, int>(5);
        matrix_serialization_test<GraphMat::CSRTile<int>, int>(500);
  }
  SECTION(" COOTile basic tests ", "COOTile basic tests") {
        matrix_serialization_test<GraphMat::COOTile<int>, int>(5);
        matrix_serialization_test<GraphMat::COOTile<int>, int>(500);
  }
  SECTION(" COOSIMD32Tile basic tests ", "COOSIMD32Tile basic tests") {
        matrix_serialization_test<GraphMat::COOSIMD32Tile<int>, int>(5);
        matrix_serialization_test<GraphMat::COOSIMD32Tile<int>, int>(500);
  }
  SECTION(" HybridTile basic tests ", "HybridTile basic tests") {
        matrix_serialization_test<GraphMat::HybridTile<int>, int>(5);
        matrix_serialization_test<GraphMat::HybridTile<int>, int>(500);
  }
}

