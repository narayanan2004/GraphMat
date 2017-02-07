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
#include "boost/serialization/vector.hpp"
#include "GraphMatRuntime.h"


class neighbors_vp {
  public: 
    int  id;
    std::vector<int> v;
  public:
    neighbors_vp() {
      id = 0;
      v = std::vector<int>(0);
    }
    int operator!=(const neighbors_vp& t) {
      return (id != t.id);
    }
    
};

class serializable_vector : public GraphMat::Serializable {
  public:
    std::vector<int> v;
  public:
    friend boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & v;
    }
};

class GetNeighbors : public GraphMat::GraphProgram<int, serializable_vector, neighbors_vp> {

  public:

  GetNeighbors() {
    this->activity = GraphMat::ALL_VERTICES;
    this->order = GraphMat::IN_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  void reduce_function(serializable_vector& a, const serializable_vector& b) const {
    //a.push_back(b.v[0]);
    a.v.insert(a.v.end(), b.v.begin(), b.v.end());
  }

  void process_message(const int& message, const int edge_val, const neighbors_vp& vertexprop, serializable_vector &res) const {
    res.v.clear();
    res.v.push_back(message);
  }

  bool send_message(const neighbors_vp& vertexprop, int& message) const {
    message = vertexprop.id; //dummy placeholder message
    return true;
  }

  void apply(const serializable_vector& message_out, neighbors_vp& vertexprop)  {
      vertexprop.v = message_out.v;
      std::sort(vertexprop.v.begin(), vertexprop.v.end());
  }

};

void test_get_neighbors(int n) {
  //auto E = generate_circular_chain_edgelist<int>(n);
  auto E = generate_dense_edgelist<int>(n);
  GraphMat::Graph<neighbors_vp> G;
  G.ReadEdgelist(E);
  E.clear();

  for (int i = 1; i <= n; i++ ) {
    if (G.vertexNodeOwner(i)) {
      auto x = G.getVertexproperty(i);
      x.id = i;
      G.setVertexproperty(i, x);
    }
  }
  GetNeighbors gn;
  auto gn_tmp = GraphMat::graph_program_init(gn, G);

  GraphMat::run_graph_program(&gn, G, 1, &gn_tmp);

  GraphMat::graph_program_clear(gn_tmp);

  std::vector<int> ref(n);
  std::iota(ref.begin(), ref.end(), 1);

  for (int i = 1; i <= n; i++) {
    if (G.vertexNodeOwner(i)) {
      //REQUIRE(G.getVertexproperty(i).v.size() == 1);
      REQUIRE(G.getVertexproperty(i).v.size() == n);
      REQUIRE(G.getVertexproperty(i).v == ref);
    }
  }
}

TEST_CASE("Get neighbors tests")
{
  SECTION("getneighbors 100") {
    test_get_neighbors(100);
  }
}

