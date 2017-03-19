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

#include "GraphMatRuntime.h"
#include <climits>
#include <ostream>

typedef unsigned int depth_type;
depth_type MAX_DIST = std::numeric_limits<depth_type>::max();

class Vertex_type {
  public: 
    depth_type topsort_order;
    int in_degree;
  public:
    Vertex_type() {
      topsort_order = MAX_DIST;
      in_degree = 0;
    }
    bool operator != (const Vertex_type& p) {
      return (this->topsort_order!= p.topsort_order);
    }

  friend std::ostream &operator<<(std::ostream &outstream, const Vertex_type & val)
    {
      outstream << val.topsort_order ;
      return outstream;
    }
};

template<class V, class E=int>
class InDegree : public GraphMat::GraphProgram<int, int, V, E> {
  public:

  InDegree() {
    this->activity = GraphMat::ALL_VERTICES;
    this->order = GraphMat::OUT_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  bool send_message(const V& vertex, int& message) const {
    message = 1;
    return true;
  }

  void process_message(const int& message, const E edge_value, const V& vertex, int& result) const {
    result = message;
  }

  void reduce_function(int& a, const int& b) const {
    a += b;
  }

  void apply(const int& message_out, V& vertex) {
    vertex.in_degree = message_out; 
  }

};


class TopSort : public GraphMat::GraphProgram<bool, int, Vertex_type> {

  public:
    depth_type current_topsort_order;
    
  public:

  TopSort() {
    current_topsort_order = 1;
    this->order = GraphMat::OUT_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  void reduce_function(int& a, const int& b) const {
    a += b;
  }

  void process_message(const bool& message, const int edge_val, const Vertex_type& vertex, int &res) const {
    res = (message == true)?(1):(0); 
    //assert(message == true);
  }

  bool send_message(const Vertex_type& vertex, bool& message) const {
    message = (vertex.in_degree == 0)?true:false;
    return true; //(vertex.in_degree == 0);
  }

  void apply(const int& message_out, Vertex_type& vertex)  {
    assert(message_out > 0);
    assert(vertex.in_degree > 0);
    vertex.in_degree -= message_out;
    if (vertex.in_degree == 0) {
      vertex.topsort_order= current_topsort_order;
    }
    assert(vertex.in_degree >= 0);
  }

  void do_every_iteration(int iteration_number) {
    current_topsort_order++;
  }

};

void unreachable(Vertex_type* v, int *result, void* params=nullptr) {
  int unreachable = 0;
  if (v->topsort_order == MAX_DIST) {
    unreachable = 1;
  } 
  *result = unreachable;
}

void run_topsort(char* filename) {
  GraphMat::Graph<Vertex_type> G;
  G.ReadMTX(filename); 
  
  InDegree<Vertex_type> indeg;
  TopSort topsort;

  auto d_tmp = GraphMat::graph_program_init(indeg, G);
  auto b_tmp = GraphMat::graph_program_init(topsort, G);


  struct timeval start, end;
  gettimeofday(&start, 0);

  GraphMat::run_graph_program(&indeg, G, 1, &d_tmp);

  G.setAllInactive();
  #pragma omp parallel for
  for (int i = 1; i <= G.getNumberOfVertices(); i++) {
    if (G.vertexNodeOwner(i)) {
      auto v = G.getVertexproperty(i);
      if (v.in_degree == 0) {
        G.setActive(i);
        v.topsort_order= 0;
        G.setVertexproperty(i, v);
      }
    }
  }
  
  GraphMat::run_graph_program(&topsort, G, GraphMat::UNTIL_CONVERGENCE, &b_tmp);

  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);
 
  GraphMat::graph_program_clear(d_tmp);
  GraphMat::graph_program_clear(b_tmp);

  int unreachable_vertices = 0;
  G.applyReduceAllVertices(&unreachable_vertices, unreachable); //default reduction = sum
  if (unreachable_vertices > 0) {
    if (GraphMat::get_global_myrank() == 0) {
      printf("Topological Sort not possible. Graph has cycles.\n");
    }
    return;
  }

  for (int i = 1; i <= std::min(10, G.getNumberOfVertices()); i++) {
    if (G.vertexNodeOwner(i)) {
      printf("Top Sort order %d : %d\n", i, G.getVertexproperty(i).topsort_order);
    }
  }

}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  if (argc < 2) {
    printf("Correct format: %s A.mtx\n", argv[0]);
    return 0;
  }

  run_topsort(argv[1]);

  MPI_Finalize();
  
}

