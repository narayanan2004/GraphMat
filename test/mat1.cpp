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



template<typename T>
bool edge_compare(const GraphPad::edge_t<T> &e1,
                  const GraphPad::edge_t<T> &e2)
{
        if( (e1.src < e2.src) ||
            ((e1.src == e2.src) && (e1.dst < e2.dst)) ||
            ((e1.src == e2.src) && (e1.dst == e2.dst) && (e1.val < e2.val)) )
        {
                return true;
        }
        return false;
}

template <typename EDGE_T>
void collect_edges(const GraphPad::edgelist_t<EDGE_T>& in_edges, GraphPad::edgelist_t<EDGE_T>& out_edges) {

    REQUIRE(sizeof(EDGE_T)%sizeof(int) == 0);
    int T_by_int = sizeof(in_edges.edges[0])/sizeof(int);

    int* OERecvCount = new int[GraphPad::get_global_nrank()];
    MPI_Allgather(&in_edges.nnz, 1, MPI_INT, OERecvCount, 1, MPI_INT, MPI_COMM_WORLD);

    int* OERecvOffset = new int[GraphPad::get_global_nrank()];
    int* OERecvCountInt = new int[GraphPad::get_global_nrank()];
    OERecvOffset[0] = 0;
    for (int i = 1; i < GraphPad::get_global_nrank(); i++) {
      OERecvOffset[i] = OERecvOffset[i-1] + T_by_int*OERecvCount[i-1];      
    }
    for (int i = 0; i < GraphPad::get_global_nrank(); i++) {
      OERecvCountInt[i] = T_by_int*OERecvCount[i];
    }

    int nnz = 0;
    for (int i = 0; i < GraphPad::get_global_nrank(); i++) {
      nnz += OERecvCount[i];
    }
    out_edges = GraphPad::edgelist_t<EDGE_T>(in_edges.m, in_edges.n, nnz);

    MPI_Allgatherv(in_edges.edges, in_edges.nnz*T_by_int, MPI_INT, out_edges.edges, OERecvCountInt, OERecvOffset, MPI_INT, MPI_COMM_WORLD);

    delete [] OERecvCount;
    delete [] OERecvCountInt;
    delete [] OERecvOffset;
}

template <typename TILE_T, typename EDGE_T>
void matrix_test(GraphPad::edgelist_t<EDGE_T> E)
{
    std::sort(E.edges, E.edges + E.nnz, edge_compare<EDGE_T>);

  // Create identity matrix from generator
    GraphPad::SpMat<TILE_T> A;
    GraphPad::AssignSpMat(E, &A, GraphPad::get_global_nrank(), GraphPad::get_global_nrank(), GraphPad::partition_fn_1d);

    //collect all edges
    GraphPad::edgelist_t<EDGE_T> EAll;
    collect_edges(E, EAll);
    std::sort(EAll.edges, EAll.edges + EAll.nnz, edge_compare<EDGE_T>);

    REQUIRE(A.getNNZ() == EAll.nnz);
    REQUIRE(A.m == E.m);
    REQUIRE(A.n == E.n);
    REQUIRE(A.empty == false);

    // Get new edgelist from matrix
    GraphPad::edgelist_t<EDGE_T> OE;
    A.get_edges(&OE);

    //collect all edges
    GraphPad::edgelist_t<EDGE_T> OEAll;
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
    GraphPad::SpMat<TILE_T> AT;
    GraphPad::Transpose(A, &AT, GraphPad::get_global_nrank(), GraphPad::get_global_nrank(), GraphPad::partition_fn_1d);
    REQUIRE(AT.getNNZ() == EAll.nnz);
    REQUIRE(AT.m == E.n);
    REQUIRE(AT.n == E.m);
    REQUIRE(AT.empty == false);

    GraphPad::SpMat<TILE_T> ATT;
    GraphPad::Transpose(AT, &ATT, GraphPad::get_global_nrank(), GraphPad::get_global_nrank(), GraphPad::partition_fn_1d);
    REQUIRE(ATT.getNNZ() == EAll.nnz);
    REQUIRE(ATT.m == E.m);
    REQUIRE(ATT.n == E.n);
    REQUIRE(ATT.empty == false);

    GraphPad::edgelist_t<EDGE_T> OET;
    ATT.get_edges(&OET);

    //collect edges
    GraphPad::edgelist_t<EDGE_T> OETAll;
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
        create_matrix_test<GraphPad::CSRTile<int>, int>(5);
        create_matrix_test<GraphPad::CSRTile<int>, int>(500);
  }
  SECTION(" DCSCTile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphPad::DCSCTile<int>, int>(5);
        create_matrix_test<GraphPad::DCSCTile<int>, int>(500);
  }
  SECTION(" COOTile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphPad::COOTile<int>, int>(5);
        create_matrix_test<GraphPad::COOTile<int>, int>(500);
  }
  SECTION(" COOSIMD32Tile basic tests ", "CSRTile basic tests") {
        create_matrix_test<GraphPad::COOSIMD32Tile<int>, int>(5);
        create_matrix_test<GraphPad::COOSIMD32Tile<int>, int>(500);
  }
}

template <typename T>
void mul(T a, T b, T * c, void* vsp) {*c = a*b;}

template <typename T>
void add(T a, T b, T * c, void* vsp) {*c = a+b;}

template <typename TILE_T, typename EDGE_T>
void spgemm_IxI_test(GraphPad::edgelist_t<EDGE_T> E, 
                     GraphPad::edgelist_t<EDGE_T> R)
{
    GraphPad::SpMat<TILE_T> A;
    GraphPad::AssignSpMat(E, &A, 1, 1, GraphPad::partition_fn_1d);

    GraphPad::SpMat<TILE_T> B;
    GraphPad::AssignSpMat(R, &B, 1, 1, GraphPad::partition_fn_1d);

    GraphPad::SpMat<TILE_T> C;
    GraphPad::SpGEMM(A, B, &C, mul, add);

    GraphPad::edgelist_t<EDGE_T> OE;
    C.get_edges(&OE);
    REQUIRE(E.nnz == OE.nnz);
    REQUIRE(E.m == OE.m);
    REQUIRE(E.n == OE.n);

    std::sort(OE.edges, OE.edges + OE.nnz, edge_compare<EDGE_T>);
    std::sort(E.edges, E.edges + E.nnz, edge_compare<EDGE_T>);

    for(int i = 0 ; i < OE.nnz ; i++)
    {
      REQUIRE(OE.edges[i].src == OE.edges[i].dst);
      REQUIRE(OE.edges[i].val == Approx(E.edges[i].val * E.edges[i].val)); 
    }
}

template <typename TILE_T, typename EDGE_T>
void create_spgemm_test(int N)
{
  auto E1 = generate_identity_edgelist<EDGE_T>(N);
  auto E2 = generate_identity_edgelist<EDGE_T>(N);
  spgemm_IxI_test<TILE_T, EDGE_T>(E1, E2);

}

TEST_CASE("spgemm", "spgemm")
{
  SECTION(" CSR SpGEMM", "CSR SpGEMM") {
    create_spgemm_test<GraphPad::CSRTile<double>, double>(50);
  }
}

template <typename TILE_T, typename EDGE_T>
void create_spmv_identity_test(int N) {
    auto E1 = generate_identity_edgelist<EDGE_T>(N);
    GraphPad::SpMat<TILE_T> A;
    GraphPad::AssignSpMat(E1, &A, GraphPad::get_global_nrank(), GraphPad::get_global_nrank(), GraphPad::partition_fn_1d);
    
    auto E2 = generate_random_vector_edgelist<EDGE_T>(N, N/10);
    GraphPad::SpVec<GraphPad::DenseSegment<EDGE_T> > x;
    x.AllocatePartitioned(N, GraphPad::get_global_nrank(), GraphPad::vector_partition_fn);
    x.ingestEdgelist(E2);

    GraphPad::SpVec<GraphPad::DenseSegment<EDGE_T> > y;
    y.AllocatePartitioned(N, GraphPad::get_global_nrank(), GraphPad::vector_partition_fn);
    GraphPad::Clear(&y);

    GraphPad::SpMSpV(A, x, &y, mul, add, NULL);

    REQUIRE(x.getNNZ() == y.getNNZ());

    GraphPad::edgelist_t<EDGE_T> E3;
    GraphPad::edgelist_t<EDGE_T> E4;
    y.get_edges(&E3);
    collect_edges(E3, E4);
    std::sort(E4.edges, E4.edges + E4.nnz, edge_compare<EDGE_T>);
    for (int i = 0; i < E3.nnz; i++) {
      printf("%d %d %lf\n", E3.edges[i].src, E3.edges[i].dst, E3.edges[i].val);
    }

    GraphPad::edgelist_t<EDGE_T> E5;
    collect_edges(E2, E5);
    std::sort(E5.edges, E5.edges + E5.nnz, edge_compare<EDGE_T>);
    for (int i = 0; i < E5.nnz; i++) {
      printf("%d %d %lf\n", E5.edges[i].src, E5.edges[i].dst, E5.edges[i].val);
    }
    for (int i = 0; i < E2.nnz; i++) {
      printf("%d %d %lf\n", E2.edges[i].src, E2.edges[i].dst, E2.edges[i].val);
    }

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
    create_spmv_identity_test<GraphPad::CSRTile<double>, double>(100);
  }
}
