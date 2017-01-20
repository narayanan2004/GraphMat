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

#include <climits>
#include <cfloat>

#include "GraphMatRuntime.h"

//typedef unsigned char distance_type;
typedef unsigned int distance_type;
//typedef double distance_type;
//typedef float distance_type;

distance_type MAX_DIST = std::numeric_limits<distance_type>::max();

class SSSP_vertex_type {
  public: 
    distance_type distance;
  public:
    SSSP_vertex_type() {
      distance = MAX_DIST;
    }
    bool operator != (const SSSP_vertex_type& p) {
      return (this->distance != p.distance);
    }
    void print() const {
      if (distance < MAX_DIST) {
        std::cout << "distance = " << distance << std::endl;
      } else {
        std::cout << "distance = INF" << std::endl;
      }
    }
};

template <class edge_type>
class SSSP : public GraphMat::GraphProgram<distance_type, distance_type, SSSP_vertex_type, edge_type> {

  public:

  SSSP() {
    this->order = GraphMat::OUT_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  void reduce_function(distance_type& a, const distance_type& b) const {
    a = (a<=b)?(a):(b);
  }

  void process_message(const distance_type& message, const edge_type edge_val, const SSSP_vertex_type& vertexprop, distance_type &res) const {
    res = message + edge_val;
  }

  bool send_message(const SSSP_vertex_type& vertexprop, distance_type& message) const {
    message = vertexprop.distance;
    return true;
  }

  void apply(const distance_type& message_out, SSSP_vertex_type& vertexprop)  {
    vertexprop.distance = std::min(vertexprop.distance, message_out);
  }

};

void reachable_or_not(SSSP_vertex_type* v, int *result, void* params=nullptr) {
  int reachable = 0;
  if (v->distance < MAX_DIST) {
    reachable = 1;
  } 
  *result = reachable;
}


template<class edge_type>
void run_sssp(const char* filename, int v) {

  GraphMat::Graph<SSSP_vertex_type, edge_type> G;
  G.ReadMTX(filename); 

  SSSP<edge_type> b;
  auto tmp_ds = GraphMat::graph_program_init(b, G);

  SSSP_vertex_type init; 
  init.distance = 0; 

  SSSP_vertex_type inf; 
  G.setAllVertexproperty(inf);
  G.setAllInactive();

  G.setVertexproperty(v, init);
  G.setActive(v);

  struct timeval start, end;
  gettimeofday(&start, 0);

  GraphMat::run_graph_program(&b, G, GraphMat::UNTIL_CONVERGENCE, &tmp_ds);

  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);
  
 
  int reachable_vertices = 0;
  G.applyReduceAllVertices(&reachable_vertices, reachable_or_not); //default reduction = sum

  if (GraphMat::get_global_myrank() == 0) printf("Reachable vertices = %d \n", reachable_vertices);

  for (int i = 1; i <= std::min((unsigned long long int)25, (unsigned long long int)G.nvertices); i++) {
    if (G.vertexNodeOwner(i)) {
      printf("%d : ", i);
      G.getVertexproperty(i).print();
    }
  }
  
  GraphMat::graph_program_clear(tmp_ds);
}

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  const char* input_filename = argv[1];

  if (argc < 3) {
    printf("Correct format: %s A.mtx source_vertex (1-based index)\n", argv[0]);
    return 0;
  }

  int source_vertex = atoi(argv[2]);
  run_sssp<int>(input_filename, source_vertex);
  MPI_Finalize();
 
}

