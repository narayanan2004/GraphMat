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

class BFSD2 {
  public: 
    depth_type depth;
    unsigned long long int parent;
    unsigned long long int id;
  public:
    BFSD2() {
      depth = MAX_DIST;
      parent = -1;
      id = -1;
    }
    bool operator != (const BFSD2& p) {
      return (this->depth != p.depth);
    }

  friend std::ostream &operator<<(std::ostream &outstream, const BFSD2 & val)
    {
      outstream << val.depth; 
      return outstream;
    }
};

class BFS2 : public GraphMat::GraphProgram<unsigned long long int, unsigned long long int, BFSD2> {

  public:
    depth_type current_depth;
    
  public:

  BFS2() {
    current_depth = 1;
    this->order = GraphMat::OUT_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  void reduce_function(unsigned long long int& a, const unsigned long long int& b) const {
    a=b;
  }

  void process_message(const unsigned long long int& message, const int edge_val, const BFSD2& vertexprop, unsigned long long int &res) const {
    res = message;
  }

  bool send_message(const BFSD2& vertexprop, unsigned long long int& message) const {
    message = vertexprop.id;
    return (vertexprop.depth == current_depth-1);
  }

  void apply(const unsigned long long int& message_out, BFSD2& vertexprop)  {
    if (vertexprop.depth == MAX_DIST) {
      vertexprop.depth = current_depth;
      vertexprop.parent = message_out;
    }
  }

  void do_every_iteration(int iteration_number) {
    current_depth++;
  }

};

void reachable_or_not(BFSD2* v, int *result, void* params=nullptr) {
  int reachable = 0;
  if (v->depth < MAX_DIST) {
    reachable = 1;
  } 
  *result = reachable;
}


void run_bfs(char* filename, int v) {
  GraphMat::Graph<BFSD2> G;
  G.ReadMTX(filename); 
  
  for(int i = 0 ; i < G.getNumberOfVertices() ; i++)
  {
    BFSD2 vp = G.getVertexproperty(i+1);
    vp.id = i+1;
    G.setVertexproperty(i+1, vp);
  }
  BFS2 b;

  auto b_tmp = GraphMat::graph_program_init(b, G);

  G.setAllInactive();

  //G.vertexproperty[v].depth = 0;
  auto source = G.getVertexproperty(v);
  source.depth = 0;
  G.setVertexproperty(v, source);
  G.setActive(v);

  struct timeval start, end;
  gettimeofday(&start, 0);

  GraphMat::run_graph_program(&b, G, GraphMat::UNTIL_CONVERGENCE, &b_tmp);

  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);
 
  GraphMat::graph_program_clear(b_tmp);

  int reachable_vertices = 0;
  G.applyReduceAllVertices(&reachable_vertices, reachable_or_not); //default reduction = sum
  if (GraphMat::get_global_myrank() == 0) printf("Reachable vertices = %d \n", reachable_vertices);

  for (int i = 1; i <= std::min(10, G.getNumberOfVertices()); i++) {
    if (G.vertexNodeOwner(i))
    if (G.getVertexproperty(i).depth < MAX_DIST) {
      printf("Depth %d : %d parent: %lld\n", i, G.getVertexproperty(i).depth, G.getVertexproperty(i).parent);
    }
    else {
      printf("Depth %d : INF \n", i);
    }
  }

}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  if (argc < 3) {
    printf("Correct format: %s A.mtx source_vertex (1-based index)\n", argv[0]);
    return 0;
  }

  int source_vertex = atoi(argv[2]);
  run_bfs(argv[1], source_vertex);

  MPI_Finalize();
  
}

