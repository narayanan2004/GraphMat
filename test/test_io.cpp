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
#include <fstream>
#include <string>
#include "catch.hpp"
#include "generator.h"
#include "test_utils.h"
#include "Graph.h"

template<typename T>
void test_read_mtx(int n, bool binaryformat, bool header, bool edgeweights) {
  auto E = generate_dense_edgelist<T>(n);

  std::string tempfilenamestr = "GM_tempfileXXXXXX" + std::to_string(GraphMat::get_global_myrank());
  int suffixlen = std::to_string(GraphMat::get_global_myrank()).size();
  char* tempfilename = new char[tempfilenamestr.size()+1];

  int fd;
  memcpy(tempfilename, tempfilenamestr.c_str(), tempfilenamestr.size()*sizeof(char));
  tempfilename[tempfilenamestr.size()] = '\0';
  fd = mkstemps(tempfilename, suffixlen);
  REQUIRE(fd != -1);
 
  char* tempfilenamewithoutsuffix = new char[tempfilenamestr.size() - suffixlen + 1];
  memcpy(tempfilenamewithoutsuffix, tempfilename, (tempfilenamestr.size() - suffixlen)*sizeof(char));
  tempfilenamewithoutsuffix[ tempfilenamestr.size() - suffixlen] = '\0';

  GraphMat::edgelist_t<T> E2;

  GraphMat::write_edgelist<T>(tempfilenamewithoutsuffix, E, binaryformat, header, edgeweights); //text format with header and edgeweights
  GraphMat::load_edgelist<T>(tempfilenamewithoutsuffix, &E2, binaryformat, header, edgeweights);

  unlink(tempfilename);
  MPI_Barrier(MPI_COMM_WORLD);

  GraphMat::edgelist_t<T> E_out;
  collect_edges(E, E_out);
  std::sort(E_out.edges, E_out.edges + E_out.nnz, edge_compare<T>);

  GraphMat::edgelist_t<T> E2_out;
  collect_edges(E2, E2_out);
  std::sort(E2_out.edges, E2_out.edges + E2_out.nnz, edge_compare<T>);

  REQUIRE(E_out.nnz == E2_out.nnz);
  for (int i = 0; i < E_out.nnz; i++) {
    REQUIRE(E_out.edges[i].src == E2_out.edges[i].src);
    REQUIRE(E_out.edges[i].dst == E2_out.edges[i].dst);
    if (edgeweights) REQUIRE(E_out.edges[i].val == E2_out.edges[i].val);
  }
  E.clear();
  E_out.clear();
  E2.clear();
  E2_out.clear();
}

template<typename T>
void test_read_gm_bin(int n) {
  auto E = generate_dense_edgelist<T>(n);
  GraphMat::edgelist_t<T> E2;

  std::string tempfilenamestr = "GM_tempfileXXXXXX" + std::to_string(GraphMat::get_global_myrank());
  int suffixlen = std::to_string(GraphMat::get_global_myrank()).size();
  char* tempfilename = new char[tempfilenamestr.size()+1];

  int fd;
  memcpy(tempfilename, tempfilenamestr.c_str(), tempfilenamestr.size()*sizeof(char));
  tempfilename[tempfilenamestr.size()] = '\0';
  fd = mkstemps(tempfilename, suffixlen);
  REQUIRE(fd != -1);
 
  char* tempfilenamewithoutsuffix = new char[tempfilenamestr.size() - suffixlen + 1];
  memcpy(tempfilenamewithoutsuffix, tempfilename, (tempfilenamestr.size() - suffixlen)*sizeof(char));
  tempfilenamewithoutsuffix[ tempfilenamestr.size() - suffixlen] = '\0';

  {
    GraphMat::Graph<int, T> G;
    G.ReadEdgelist(E);
    G.WriteGraphMatBin(tempfilenamewithoutsuffix);
  }


  {
    GraphMat::Graph<double, T> G2;
    G2.ReadGraphMatBin(tempfilenamewithoutsuffix);
    G2.getEdgelist(E2); 
  }

  unlink(tempfilename);

  GraphMat::edgelist_t<T> E_out;
  collect_edges(E, E_out);
  std::sort(E_out.edges, E_out.edges + E_out.nnz, edge_compare<T>);

  GraphMat::edgelist_t<T> E2_out;
  collect_edges(E2, E2_out);
  std::sort(E2_out.edges, E2_out.edges + E2_out.nnz, edge_compare<T>);

  REQUIRE(E_out.nnz == E2_out.nnz);
  for (int i = 0; i < E_out.nnz; i++) {
    REQUIRE(E_out.edges[i].src == E2_out.edges[i].src);
    REQUIRE(E_out.edges[i].dst == E2_out.edges[i].dst);
    REQUIRE(E_out.edges[i].val == E2_out.edges[i].val);
  }
  E.clear();
  E_out.clear();
  E2.clear();
  E2_out.clear();
}

TEST_CASE("IO") 
{
  SECTION("Test file IO (int mtx)") {
    test_read_mtx<int>(10, true, true, true);
    test_read_mtx<int>(10, true, true, false);
    test_read_mtx<int>(10, true, false, true);
    test_read_mtx<int>(10, true, false, false);
    test_read_mtx<int>(10, false, true, true);
    test_read_mtx<int>(10, false, true, false);
    test_read_mtx<int>(10, false, false, true);
    test_read_mtx<int>(10, false, false, false);
  }
  SECTION("Test file IO (float mtx)") {
    test_read_mtx<float>(10, true, true, true);
    test_read_mtx<float>(10, true, true, false);
    test_read_mtx<float>(10, true, false, true);
    test_read_mtx<float>(10, true, false, false);
    test_read_mtx<float>(10, false, true, true);
    test_read_mtx<float>(10, false, true, false);
    test_read_mtx<float>(10, false, false, true);
    test_read_mtx<float>(10, false, false, false);
  }
  SECTION("Test file IO (GM bin)") {
    test_read_gm_bin<int>(10);
    test_read_gm_bin<float>(10);
  }
}

