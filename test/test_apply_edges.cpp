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
#include "Graph.h"

void apply_edges_fn(int * edge_val, int dst_vp, int src_vp, void * vsp)
{
  *edge_val = dst_vp + 2*src_vp;
}

template <typename TILE_T, typename EDGE_T>
void apply_edges(GraphMat::edgelist_t<EDGE_T> E)
{

    GraphMat::Graph<int, EDGE_T> G;
    G.ReadEdgelist(E);
    for(int i = 1 ; i <= G.getNumberOfVertices() ; i++)
    {
      G.setVertexproperty(i, i);
    }

    // Apply edges
    G.applyToAllEdges(apply_edges_fn, NULL);
    GraphMat::edgelist_t<EDGE_T> E2;
    G.getEdgelist(E2);
    for(int i = 0 ; i < E2.nnz ; i++)
    {
            REQUIRE(E2.edges[i].val == E2.edges[i].src + E2.edges[i].dst);
    }
    E2.clear();


    std::sort(E.edges, E.edges + E.nnz, edge_compare<EDGE_T>);
    GraphMat::SpMat<TILE_T>* A = new GraphMat::SpMat<TILE_T>(E, GraphMat::get_global_nrank(), GraphMat::get_global_nrank(), GraphMat::partition_fn_1d);
    GraphMat::SpVec<GraphMat::DenseSegment<EDGE_T> > vp(A->n, GraphMat::get_global_nrank(), GraphMat::vector_partition_fn);
    for(int i = 0 ; i < A->n ; i++)
    {
      vp.set(i+1, i+1);
    }

    // Apply edges
    GraphMat::ApplyEdges(A, &vp, apply_edges_fn, NULL);

    // Get new edgelist from matrix
    GraphMat::edgelist_t<EDGE_T> OE;
    std::cout << "Getting edges first" << std::endl;
    A->get_edges(&OE);

    for(int i = 0 ; i < OE.nnz ; i++)
    {
            REQUIRE(OE.edges[i].val == 2*OE.edges[i].src + OE.edges[i].dst);
    }

    delete A;
    E.clear();
    OE.clear();
}

template <typename TILE_T, typename EDGE_T>
void create_matrix_apply_edges(int N)
{
  auto E = generate_identity_edgelist<EDGE_T>(N);
  apply_edges<TILE_T, EDGE_T>(E);

  auto R = generate_random_edgelist<EDGE_T>(N, 16);
  apply_edges<TILE_T, EDGE_T>(R);
}


TEST_CASE("apply_edges_test", "apply_edges_test")
{
  SECTION(" DCSCTile apply edges", "DCSCTile apply edges") {
        create_matrix_apply_edges<GraphMat::DCSCTile<int>, int>(5);
        create_matrix_apply_edges<GraphMat::DCSCTile<int>, int>(500);
  }
}

