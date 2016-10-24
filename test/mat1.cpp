
#include <iostream>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "generator.hpp"

int main(int argc, char * argv[])
{
  MPI_Init(NULL,NULL);
  GraphPad::GB_Init();

  int res =  Catch::Session().run(argc, argv);

  MPI_Finalize();
  return res;
}

template <typename TILE_T, typename EDGE_T>
void identity_nnz_test(GraphPad::edgelist_t<EDGE_T> E)
{
  // Create identity matrix from generator
    GraphPad::SpMat<TILE_T> A;
    GraphPad::AssignSpMat(E, &A, 1, 1, GraphPad::partition_fn_1d);

    REQUIRE(A.getNNZ() == E.nnz);
    REQUIRE(A.m == E.m);
    REQUIRE(A.n == E.n);
    REQUIRE(A.empty == false);

    // Get new edgelist from matrix
    GraphPad::edgelist_t<EDGE_T> OE;
    A.get_edges(&OE);

    REQUIRE(E.nnz == OE.nnz);
    REQUIRE(E.m == OE.m);
    REQUIRE(E.n == OE.n);
    for(int i = 0 ; i < E.nnz ; i++)
    {
            REQUIRE(E.edges[i].src == OE.edges[i].src);
            REQUIRE(E.edges[i].dst == OE.edges[i].dst);
            REQUIRE(E.edges[i].val == OE.edges[i].val);
    }

    // Test transpose
    GraphPad::SpMat<TILE_T> AT;
    GraphPad::Transpose(A, &AT, 1, 1, GraphPad::partition_fn_1d);
    REQUIRE(AT.getNNZ() == E.nnz);
    REQUIRE(AT.m == E.n);
    REQUIRE(AT.n == E.m);
    REQUIRE(AT.empty == false);

    GraphPad::SpMat<TILE_T> ATT;
    GraphPad::Transpose(AT, &ATT, 1, 1, GraphPad::partition_fn_1d);
    REQUIRE(ATT.getNNZ() == E.nnz);
    REQUIRE(ATT.m == E.m);
    REQUIRE(ATT.n == E.n);
    REQUIRE(ATT.empty == false);

    GraphPad::edgelist_t<EDGE_T> OET;
    ATT.get_edges(&OET);

    REQUIRE(E.nnz == OET.nnz);
    REQUIRE(E.m == OET.m);
    REQUIRE(E.n == OET.n);
    for(int i = 0 ; i < E.nnz ; i++)
    {
            REQUIRE(E.edges[i].src == OET.edges[i].src);
            REQUIRE(E.edges[i].dst == OET.edges[i].dst);
            REQUIRE(E.edges[i].val == OET.edges[i].val);
    }
}

template <typename TILE_T, typename EDGE_T>
void nnz_test(int N)
{
  auto E = generate_identity_edgelist<EDGE_T>(N);
  identity_nnz_test<TILE_T, EDGE_T>(E);
}


TEST_CASE("identity_nnz", "identity_nnz")
{
  SECTION(" CSRTile basic tests ", "CSRTile basic tests") {
        nnz_test<GraphPad::CSRTile<int>, int>(500);
  }
  SECTION(" DCSCTile basic tests ", "CSRTile basic tests") {
        nnz_test<GraphPad::DCSCTile<int>, int>(500);
  }
  SECTION(" COOTile basic tests ", "CSRTile basic tests") {
        nnz_test<GraphPad::COOTile<int>, int>(500);
  }
}


