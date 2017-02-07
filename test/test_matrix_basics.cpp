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
void matrix_test(GraphMat::edgelist_t<EDGE_T> E)
{
    std::sort(E.edges, E.edges + E.nnz, edge_compare<EDGE_T>);

  // Create identity matrix from generator
    GraphMat::SpMat<TILE_T>* A = new GraphMat::SpMat<TILE_T>(E, GraphMat::get_global_nrank(), GraphMat::get_global_nrank(), GraphMat::partition_fn_1d);
    //GraphMat::SpMat<TILE_T> A(E, GraphMat::get_global_nrank(), GraphMat::get_global_nrank(), GraphMat::partition_fn_1d);

    //collect all edges
    GraphMat::edgelist_t<EDGE_T> EAll;
    collect_edges(E, EAll);
    std::sort(EAll.edges, EAll.edges + EAll.nnz, edge_compare<EDGE_T>);

    REQUIRE(A->getNNZ() == EAll.nnz);
    REQUIRE(A->m == E.m);
    REQUIRE(A->n == E.n);

    // Get new edgelist from matrix
    GraphMat::edgelist_t<EDGE_T> OE;
    std::cout << "Getting edges first" << std::endl;
    A->get_edges(&OE);

    //collect all edges
    GraphMat::edgelist_t<EDGE_T> OEAll;
    collect_edges(OE, OEAll);
    std::sort(OEAll.edges, OEAll.edges + OEAll.nnz, edge_compare<EDGE_T>);

    REQUIRE(EAll.nnz == OEAll.nnz);
    REQUIRE(EAll.m == OEAll.m);
    REQUIRE(EAll.n == OEAll.n);
    for(int i = 0 ; i < EAll.nnz ; i++)
    {
            REQUIRE(EAll.edges[i].src == OEAll.edges[i].src);
            REQUIRE(EAll.edges[i].dst == OEAll.edges[i].dst);
            REQUIRE(EAll.edges[i].val == OEAll.edges[i].val);
    }

    // Test transpose
    GraphMat::SpMat<TILE_T>* AT;
    GraphMat::Transpose(A, &AT, GraphMat::get_global_nrank(), GraphMat::get_global_nrank(), GraphMat::partition_fn_1d);
    REQUIRE(AT->getNNZ() == EAll.nnz);
    REQUIRE(AT->m == E.n);
    REQUIRE(AT->n == E.m);

    GraphMat::SpMat<TILE_T> *ATT;
    GraphMat::Transpose(AT, &ATT, GraphMat::get_global_nrank(), GraphMat::get_global_nrank(), GraphMat::partition_fn_1d);
    REQUIRE(ATT->getNNZ() == EAll.nnz);
    REQUIRE(ATT->m == E.m);
    REQUIRE(ATT->n == E.n);

    GraphMat::edgelist_t<EDGE_T> OET;
    std::cout << "Getting edges second" << std::endl;
    ATT->get_edges(&OET);

    //collect edges
    GraphMat::edgelist_t<EDGE_T> OETAll;
    collect_edges(OET, OETAll);
    std::sort(OETAll.edges, OETAll.edges + OETAll.nnz, edge_compare<EDGE_T>);

    REQUIRE(EAll.nnz == OETAll.nnz);
    REQUIRE(E.m == OET.m);
    REQUIRE(E.n == OET.n);
    for(int i = 0 ; i < EAll.nnz ; i++)
    {
            REQUIRE(EAll.edges[i].src == OETAll.edges[i].src);
            REQUIRE(EAll.edges[i].dst == OETAll.edges[i].dst);
            REQUIRE(EAll.edges[i].val == OETAll.edges[i].val);
    }
    delete A;
    delete AT;
    delete ATT;
    E.clear();
    OE.clear();
    EAll.clear();
    OEAll.clear();
    OET.clear();
    OETAll.clear();
}

template <typename TILE_T, typename EDGE_T>
void create_matrix_test(int N)
{
  auto E = generate_identity_edgelist<EDGE_T>(N);
  matrix_test<TILE_T, EDGE_T>(E);

  auto R = generate_random_edgelist<EDGE_T>(N, 16);
  matrix_test<TILE_T, EDGE_T>(R);
}


TEST_CASE("matrix_nnz", "matrix_nnz")
{
  SECTION(" CSRTile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphMat::CSRTile<int>, int>(5);
        create_matrix_test<GraphMat::CSRTile<int>, int>(500);
  }
  SECTION(" DCSCTile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphMat::DCSCTile<int>, int>(5);
        create_matrix_test<GraphMat::DCSCTile<int>, int>(500);
  }
  SECTION(" COOTile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphMat::COOTile<int>, int>(5);
        create_matrix_test<GraphMat::COOTile<int>, int>(500);
  }
  SECTION(" COOSIMD32Tile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphMat::COOSIMD32Tile<int>, int>(5);
        create_matrix_test<GraphMat::COOSIMD32Tile<int>, int>(500);
  }
  SECTION(" DCSRTile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphMat::DCSRTile<int>, int>(5);
        create_matrix_test<GraphMat::DCSRTile<int>, int>(500);
  }
  SECTION(" HybridTile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphMat::HybridTile<int>, int>(5);
        create_matrix_test<GraphMat::HybridTile<int>, int>(500);
  }
}

