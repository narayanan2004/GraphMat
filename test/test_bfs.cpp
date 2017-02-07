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

#include "catch.hpp"
#include "generator.h"
#include <algorithm>
#include <climits>
#include "GraphMatRuntime.h"

typedef unsigned int depth_type;
depth_type MAX_DIST = std::numeric_limits<depth_type>::max();

class BFSD {
  public: 
    depth_type depth;
  public:
    BFSD() {
      depth = MAX_DIST;
    }
    bool operator != (const BFSD& p) {
      return (this->depth != p.depth);
    }
  friend std::ostream &operator<<(std::ostream &outstream, const BFSD & val)
    {
      outstream << val.depth; 
      return outstream;
    }
};

class BFS : public GraphMat::GraphProgram<unsigned long long int, unsigned long long int, BFSD> {

  public:
    depth_type current_depth;
    
  public:

  BFS() {
    current_depth = 1;
    this->order = GraphMat::OUT_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  void reduce_function(unsigned long long int& a, const unsigned long long int& b) const {
    a=b;
  }

  void process_message(const unsigned long long int& message, const int edge_val, const BFSD& vertexprop, unsigned long long int &res) const {
    res = message;
  }

  bool send_message(const BFSD& vertexprop, unsigned long long int& message) const {
    message = 0; //dummy placeholder message
    return (vertexprop.depth == current_depth-1);
  }

  void apply(const unsigned long long int& message_out, BFSD& vertexprop)  {
    if (vertexprop.depth == MAX_DIST) {
      vertexprop.depth = current_depth;
    }
  }

  void do_every_iteration(int iteration_number) {
    current_depth++;
  }

};


void test_ut_bfs(int n) {
  auto E = generate_upper_triangular_edgelist<int>(n);
  GraphMat::Graph<BFSD> G;
  G.ReadEdgelist(E);
  E.clear();

  BFS bfs_program;

  SECTION ("Running BFS on node 1") {
    BFSD v;
    v.depth = 0;
    G.setVertexproperty(1, v);
    G.setAllInactive();
    G.setActive(1);

    GraphMat::run_graph_program(&bfs_program, G, GraphMat::UNTIL_CONVERGENCE); 

    if (G.vertexNodeOwner(1)) 
      REQUIRE(G.getVertexproperty(1).depth == 0);
    for (int i = 2; i <= n; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == 1);
    }
  }

  SECTION ("Running BFS on node n/2") {
    BFSD v;
    v.depth = 0;
    G.setVertexproperty(n/2, v);
    G.setAllInactive();
    G.setActive(n/2);

    GraphMat::run_graph_program(&bfs_program, G, GraphMat::UNTIL_CONVERGENCE); 

    for (int i = 1; i < n/2; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == MAX_DIST);
    }
    if (G.vertexNodeOwner(n/2)) 
      REQUIRE(G.getVertexproperty(n/2).depth == 0);
    for (int i = n/2 + 1; i <= n; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == 1);
    }
  }
}

void test_dense_bfs(int n) {
  auto E = generate_dense_edgelist<int>(n);
  GraphMat::Graph<BFSD> G;
  G.ReadEdgelist(E);
  E.clear();

  BFS bfs_program;

  SECTION ("Running BFS on node 1") {
    BFSD v;
    v.depth = 0;
    G.setVertexproperty(1, v);
    G.setAllInactive();
    G.setActive(1);

    GraphMat::run_graph_program(&bfs_program, G, GraphMat::UNTIL_CONVERGENCE); 

    if (G.vertexNodeOwner(1)) 
      REQUIRE(G.getVertexproperty(1).depth == 0);
    for (int i = 2; i <= n; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == 1);
    }
  }

  SECTION ("Running BFS on node n/2") {
    BFSD v;
    v.depth = 0;
    G.setVertexproperty(n/2, v);
    G.setAllInactive();
    G.setActive(n/2);

    GraphMat::run_graph_program(&bfs_program, G, GraphMat::UNTIL_CONVERGENCE); 

    for (int i = 1; i < n/2; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == 1);
    }
    if (G.vertexNodeOwner(n/2)) 
      REQUIRE(G.getVertexproperty(n/2).depth == 0);
    for (int i = n/2 + 1; i <= n; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == 1);
    }
  }
}

void test_chain_bfs(int n) {
  auto E = generate_circular_chain_edgelist<int>(n);
  GraphMat::Graph<BFSD> G;
  G.ReadEdgelist(E);
  E.clear();

  BFS bfs_program;

  SECTION ("Running BFS on node 1") {
    BFSD v;
    v.depth = 0;
    G.setVertexproperty(1, v);
    G.setAllInactive();
    G.setActive(1);

    GraphMat::run_graph_program(&bfs_program, G, GraphMat::UNTIL_CONVERGENCE); 

    if (G.vertexNodeOwner(1)) 
      REQUIRE(G.getVertexproperty(1).depth == 0);
    for (int i = 2; i <= n; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == i-1);
    }
  }

  SECTION ("Running BFS on node n/2") {
    BFSD v;
    v.depth = 0;
    G.setVertexproperty(n/2, v);
    G.setAllInactive();
    G.setActive(n/2);

    GraphMat::run_graph_program(&bfs_program, G, GraphMat::UNTIL_CONVERGENCE); 

    for (int i = 1; i < n/2; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == n/2+i);
    }
    if (G.vertexNodeOwner(n/2)) 
      REQUIRE(G.getVertexproperty(n/2).depth == 0);
    for (int i = n/2 + 1; i <= n; i++) {
      if (G.vertexNodeOwner(i)) 
        REQUIRE(G.getVertexproperty(i).depth == (i-n/2));
    }
  }
}

TEST_CASE("BFS tests", "[bfs][uppertriangular][dense]")
{
  SECTION("BFS upper triangular size 500") {
    test_ut_bfs(500);
  }
  SECTION("BFS upper triangular size 100") {
    test_ut_bfs(100);
  }
  SECTION("BFS dense size 500") {
    test_dense_bfs(500);
  }
  SECTION("BFS dense size 100") {
    test_dense_bfs(100);
  }
  SECTION("BFS chain size 500") {
    test_chain_bfs(500);
  }
  SECTION("BFS chain size 100") {
    test_chain_bfs(100);
  }
}
