
#include <iostream>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "generator.hpp"

TEST_CASE("identity_nnz", "identity_nnz")
{
  // Create identity matrix from generator
  MPI_Init(NULL,NULL);
  GraphPad::GB_Init();
  auto E = generate_identity_edgelist<int>(50);
  GraphPad::SpMat<GraphPad::CSRTile<int> > A;
  GraphPad::AssignSpMat(E, &A, 1, 1, GraphPad::partition_fn_1d);
  REQUIRE(A.getNNZ() == 50);
}

