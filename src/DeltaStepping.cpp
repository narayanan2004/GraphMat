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

distance_type MAX_DIST = std::numeric_limits<distance_type>::max();

class DeltaSteppingDS {
  public: 
    distance_type distance;
    int bucket;

  public:
    DeltaSteppingDS() {
      distance = MAX_DIST;
      bucket = INT_MAX;
    }
    bool operator != (const DeltaSteppingDS& p) {
      return (this->distance != p.distance);
    }
    void print() {
      if (distance < MAX_DIST) 
        printf("distance %u\n", distance);
      else
        printf("distance INF\n");
    }
};

class DeltaStepping : public GraphMat::GraphProgram<distance_type, distance_type, DeltaSteppingDS> {

  public:
    int delta;
    int bid;

  public:

  DeltaStepping(int d) {
    delta = d;
    bid = 0;
    this->order = GraphMat::OUT_EDGES;
    this->process_message_requires_vertexprop = false;
  }

  void reduce_function(distance_type& a, const distance_type& b) const {
    a = (a <= b)?(a):(b);
  }

  void process_message(const distance_type& message, const int edge_val, const DeltaSteppingDS& vertexprop, distance_type& res) const {
    res = message + edge_val; //always <= delta
  }

  bool send_message(const DeltaSteppingDS& vertexprop, distance_type& message) const {
    message = vertexprop.distance;
    return (vertexprop.bucket == bid);
  }

  void apply(const distance_type& message_out, DeltaSteppingDS& vertexprop)  {
    if (vertexprop.distance > message_out) {
      vertexprop.distance = message_out;
      vertexprop.bucket = (int)(message_out/delta);
    }
  }

};

void reachable_or_not(DeltaSteppingDS* v, int *result, void* params=nullptr) {
  int reachable = 0;
  if (v->distance < MAX_DIST) {
    reachable = 1;
  } 
  *result = reachable;
}

bool less_than_delta(GraphMat::edge_t<int> e, void* param) {
  return (e.val <= *(int*)(param));
}
bool greater_than_delta(GraphMat::edge_t<int> e, void* param) {
  return (e.val > *(int*)(param));
}

void run_deltastepping(char* filename, int delta, int source) {

  GraphMat::edgelist_t<int> E;
  GraphMat::load_edgelist<int>(filename, &E);

  auto light_edges = GraphMat::filter_edges(&E, less_than_delta, &delta);
  auto heavy_edges = GraphMat::filter_edges(&E, greater_than_delta, &delta);
  E.clear();
   
  GraphMat::Graph<DeltaSteppingDS> G;
  G.ReadEdgelist(light_edges); 
  
  GraphMat::Graph<DeltaSteppingDS> G2;
  G2.ReadEdgelist(heavy_edges); 

  light_edges.clear();
  heavy_edges.clear();

  G2.shareVertexProperty(G);

  DeltaStepping deltastep(delta);

  auto ds_ts = GraphMat::graph_program_init(deltastep, G);

  G.setAllInactive();
  DeltaSteppingDS v;
  v.distance = 0;
  v.bucket = 0;
  G.setVertexproperty(source, v);
  G.setActive(source);

  int bucket_not_empty = 1;

  struct timeval start, end;
  struct timeval start2, end2;
  double active_time = 0;

  gettimeofday(&start, 0);

  do {

  printf("***Running Bucket %d ***\n", deltastep.bid);

  GraphMat::run_graph_program(&deltastep, G, GraphMat::UNTIL_CONVERGENCE, &ds_ts);

  gettimeofday(&start2, 0);
 
  #pragma omp parallel for 
  for (int i = 1; i <= G.getNumberOfVertices(); i++) {
    if (G.vertexNodeOwner(i)) {
      if(G.getVertexproperty(i).bucket == deltastep.bid) {
        G2.setActive(i);
      } else {
        G2.setInactive(i);
      }
    }
  }
  gettimeofday(&end2, 0);
  active_time += (end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_usec-start2.tv_usec)*1e-3;

  GraphMat::run_graph_program(&deltastep, G2, 1, &ds_ts);
  deltastep.bid++;
  bucket_not_empty = 0;

  gettimeofday(&start2, 0);
  
  #pragma omp parallel for reduction(+:bucket_not_empty)
  for (int i = 1; i <= G.getNumberOfVertices(); i++) {
    if (G.vertexNodeOwner(i)) {
      if(G.getVertexproperty(i).bucket == deltastep.bid) {
        G.setActive(i);
      } else {
        G.setInactive(i);
      }
      if(G.getVertexproperty(i).bucket >= deltastep.bid && G.getVertexproperty(i).bucket < INT_MAX) {
        bucket_not_empty = 1;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &bucket_not_empty, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  gettimeofday(&end2, 0);
  active_time += (end2.tv_sec-start2.tv_sec)*1e3+(end2.tv_usec-start2.tv_usec)*1e-3;

  } while(bucket_not_empty);

  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);
  printf("Active setting time = %.3f ms \n", active_time);

  GraphMat::graph_program_clear(ds_ts);
 
  int reachable_vertices = 0;
  G.applyReduceAllVertices(&reachable_vertices, reachable_or_not); //default reduction = sum
  if (GraphMat::get_global_myrank() == 0) printf("Reachable vertices = %d \n", reachable_vertices);


  for (int i = 1; i <= std::min((unsigned long long int)25, (unsigned long long int)G.nvertices); i++) {
    if (G.vertexNodeOwner(i)) {
      printf("%d : ", i);
      G.getVertexproperty(i).print();
    }
  }
 
}

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  if (argc < 4) {
    printf("Correct format: %s A.mtx delta source\n", argv[0]);
    return 0;
  }

  int delta = atoi(argv[2]); 
  int source = atoi(argv[3]); 

  run_deltastepping(argv[1], delta, source);

  MPI_Finalize();
}

