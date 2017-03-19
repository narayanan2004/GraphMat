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
/* Narayanan Sundaram (Intel Corp.)
 * ******************************************************************************/

#include <iostream>
#include "catch.hpp"
#include "generator.h"
#include "test_utils.h"

template<typename T>
void test_remove_selfedges(int n) {

  auto E = generate_dense_edgelist<T>(n);
  GraphMat::remove_selfedges(&E);
  GraphMat::edgelist_t<T> E_out;
  collect_edges(E, E_out);

  REQUIRE(E_out.nnz == n*(n-1));
  for (int i = 0; i < E_out.nnz; i++) {
    REQUIRE(E_out.edges[i].src != E_out.edges[i].dst);
  }
  E.clear();
  E_out.clear();
}

template <typename T>
void test_remove_duplicate_edges(int n) {
  auto E = generate_dense_edgelist<T>(n);
  auto nnz = E.nnz;
  GraphMat::create_bidirectional_edges(&E);
  REQUIRE(E.nnz == 2*nnz);

  GraphMat::remove_duplicate_edges(&E);
  GraphMat::edgelist_t<T> E_out;
  collect_edges(E, E_out);

  REQUIRE(E_out.nnz == n*n);

  E.clear();
  E_out.clear();

}

template<typename T>
void test_randomize_edge_direction(int n) {

  auto E = generate_dense_edgelist<T>(n);
  GraphMat::randomize_edge_direction(&E);
  GraphMat::edgelist_t<T> E_out;
  collect_edges(E, E_out);

  REQUIRE(E_out.nnz == n*n);
  E.clear();
  E_out.clear();
}

template<typename T>
void test_bidirectional_edges(int n) {

  auto E = generate_dense_edgelist<T>(n);
  GraphMat::create_bidirectional_edges(&E);
  GraphMat::edgelist_t<T> E_out;
  collect_edges(E, E_out);

  REQUIRE(E_out.nnz == 2*n*n);
  std::sort(E_out.edges, E_out.edges + E_out.nnz, edge_compare<T>);
  for (int i = 0; i < E_out.nnz/2; i++) {
    REQUIRE(E_out.edges[2*i].src == E_out.edges[2*i+1].src);
    REQUIRE(E_out.edges[2*i].dst == E_out.edges[2*i+1].dst);
    REQUIRE(E_out.edges[2*i].val == E_out.edges[2*i+1].val);
  }
  E.clear();
  E_out.clear();
}

template<typename T>
void test_convert_to_dag(int n) {

  auto E = generate_dense_edgelist<T>(n);
  GraphMat::convert_to_dag(&E);
  GraphMat::edgelist_t<T> E_out;
  collect_edges(E, E_out);

  REQUIRE(E_out.nnz == n*n);
  for (int i = 0; i < E_out.nnz; i++) {
    REQUIRE(E_out.edges[i].src <= E_out.edges[i].dst);
  }
  E.clear();
  E_out.clear();
}

template<typename T>
bool src_div_by_3(GraphMat::edge_t<T> e, void* param) {
  return (e.src % 3 == 0);
}

template<typename T>
void test_filter_edges(int n) {
  auto E = generate_dense_edgelist<T>(n);
  auto E_out = GraphMat::filter_edges(&E, src_div_by_3<T>, nullptr);

  REQUIRE(E_out.nnz <= n*n);
  std::sort(E_out.edges, E_out.edges + E_out.nnz, edge_compare<T>);
  std::sort(E.edges, E.edges + E.nnz, edge_compare<T>);

  int k = 0;
  for (int i = 0; i < E.nnz; i++) {
    if (E.edges[i].src % 3 == 0) {
      REQUIRE(E_out.edges[k].src % 3 == 0);
      k++;
    }
  }
  REQUIRE(E_out.nnz == k);

  E.clear();
  E_out.clear();
}

TEST_CASE("edgelist transformations") 
{
  SECTION("Test remove self edges") {
    test_remove_selfedges<int>(5);
    test_remove_selfedges<float>(100);
  }
  SECTION("Test remove duplicate edges") {
    test_remove_duplicate_edges<int>(5);
    test_remove_duplicate_edges<float>(100);
  }
  SECTION("Test create bidirectional edges") {
    test_bidirectional_edges<int>(5);
    test_bidirectional_edges<float>(100);
  }
  SECTION("Test randomize edge direction") {
    test_randomize_edge_direction<int>(5);
    test_randomize_edge_direction<float>(100);
  }
  SECTION("Test convert to dag") {
    test_convert_to_dag<int>(5);
    test_convert_to_dag<float>(100);
  }
  SECTION("Test filter edges") {
    test_filter_edges<int>(5);
    test_filter_edges<float>(100);
  }
}
