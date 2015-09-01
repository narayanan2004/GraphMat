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
#include "GraphMatRuntime.cpp"

//typedef unsigned char distance_type;
//typedef unsigned int distance_type;
typedef double distance_type;

//distance_type MAX_DIST = 255;
//distance_type MAX_DIST = INT_MAX;
distance_type MAX_DIST = DBL_MAX;


class DeltaSteppingDS {
  public: 
    distance_type distance;
    int bucket;
    long long int parent;
    long long int id;

  public:
    DeltaSteppingDS() {
      distance = MAX_DIST;
      parent = -1;
      id = -1;
      bucket = INT_MAX;
    }
    bool operator != (const DeltaSteppingDS& p) {
      return (this->distance != p.distance);
    }
    void print() {
      if (distance < MAX_DIST) 
        //printf("id %d\t distance %d\t parent %d\n", id, distance, parent);
        printf("id %lld\t distance %.1f\t parent %lld\n", id, distance, parent);
      else
        printf("id %lld\t distance INF\t parent --\n", id);
    }
};

class ID_dist {
  public:
    distance_type distance;
    long long int id;
};

//class LightDeltaStepping : public GraphProgram<ID_dist, ID_dist, DeltaSteppingDS> {
class DeltaStepping : public GraphProgram<ID_dist, ID_dist, DeltaSteppingDS> {

  public:
    int delta;
    int bid;

  public:

  DeltaStepping(int d) {
    delta = d;
    bid = 0;
    this->order = OUT_EDGES;
    //this->order = ALL_EDGES;
  }

  void reduce_function(ID_dist& a, const ID_dist& b) const {
    a = (a.distance<=b.distance)?(a):(b);
  }

  void process_message(const ID_dist& message, const int edge_val, const DeltaSteppingDS& vertexprop, ID_dist& res) const {
    res = message;
    //distance_type d = (edge_val<=delta)?(res.distance+edge_val):MAX_DIST;
    //res.distance = d;
    res.distance += edge_val; //always <= delta
  }

  bool send_message(const DeltaSteppingDS& vertexprop, ID_dist& message) const {
    message.distance = vertexprop.distance;
    message.id = vertexprop.id;
    return (vertexprop.bucket == bid);
  }

  void apply(const ID_dist& message_out, DeltaSteppingDS& vertexprop)  {
    //vertexprop.depth = std::min(vertexprop.depth, message_out);
    if (vertexprop.distance > message_out.distance) {
      vertexprop.distance = message_out.distance;
      vertexprop.parent = message_out.id;
      vertexprop.bucket = (int)(message_out.distance/delta);
    }
  }

  void do_every_iteration(int iteration_number) {
    //current_depth++;
  }

};

/*
class HeavyDeltaStepping : public GraphProgram<DeltaSteppingDS, DeltaSteppingDS, DeltaSteppingDS> {

  public:
    int delta;

  public:

  HeavyDeltaStepping(int d) {
    delta = d;
    //this->order = OUT_EDGES;
    this->order = ALL_EDGES;
  }

  void reduce_function(DeltaSteppingDS& a, const DeltaSteppingDS& b) const {
    a = (a.distance<=b.distance)?(a):(b);
  }

  void process_message(const DeltaSteppingDS& message, const int edge_val, const DeltaSteppingDS& vertexprop, DeltaSteppingDS& res) const {
    res = message;
    int d = (edge_val>delta)?(res.distance+edge_val):INT_MAX;
    res.distance = d;
  }

  bool send_message(const DeltaSteppingDS& vertexprop, DeltaSteppingDS& message) const {
    message = vertexprop;
    return true;
  }

  void apply(const DeltaSteppingDS& message_out, DeltaSteppingDS& vertexprop)  {
    //vertexprop.depth = std::min(vertexprop.depth, message_out);
    if (vertexprop.distance > message_out.distance) {
      vertexprop.distance = message_out.distance;
      vertexprop.parent = message_out.id;
      vertexprop.bucket = (int)(message_out.distance/delta);
    }
  }

  void do_every_iteration(int iteration_number) {
    //current_depth++;
  }

};
*/

extern unsigned long long int edges_traversed;

void run_deltastepping(char* light_filename, char* heavy_filename, int delta, int nthreads) {
  //Graph<BFSD> G;
  //Graph<BFSD2> G;
  Graph<DeltaSteppingDS> G;
  G.ReadMTX(light_filename, nthreads*4); //nthread pieces of matrix
  
  Graph<DeltaSteppingDS> G2;
  G2.ReadMTX(heavy_filename, nthreads*4); //nthread pieces of matrix

  //delete [] G2.vertexproperty;
  //G2.vertexproperty = G.vertexproperty;
  G2.shareVertexProperty(G);

  //BFS b;
  //BFS2 b;
  //SSSP b;
  //int delta = 8;
  DeltaStepping deltastep(delta);
  //LightDeltaStepping lds(delta);
  //HeavyDeltaStepping hds(delta);
  
  for (int i = 0; i <= G.nvertices; i++) 
    G.vertexproperty[i].id = i;

  //int* S = new int[(g.nvertices+1+31)/32];
  //bool *S = new bool[G.nvertices+1];
  //memset(S, 0, sizeof(bool)*(G.nvertices+1));
  auto ds_ts = graph_program_init(deltastep, G);
  //auto lds_ts = graph_program_init(lds, G);
  //auto hds_ts = graph_program_init(hds, G2);
  

  int v = 5;
  //int v = 1758293;
  G.vertexproperty[v].distance = 0;
  G.vertexproperty[v].bucket = 0;
  G.active[v] = true;

  int bucket_not_empty = 1;
  int* next_bucket = new int[nthreads*16];

  struct timeval start, end;
  unsigned long long int start_active, end_active, cycles_active;
  cycles_active = 0;
  gettimeofday(&start, 0);

  do {

  printf("***Running Bucket %d ***\n", deltastep.bid);

  run_graph_program(&deltastep, G, -1, &ds_ts);

  start_active = __rdtsc(); 
  //memset(G2.active, 0xff, sizeof(bool)*(G2.nvertices+1));
  //memset(G2.active, 0, sizeof(bool)*(G2.nvertices+1));
 
  #pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < G.nvertices; i++) {
    if(G.vertexproperty[i].bucket == deltastep.bid) {
      G2.active[i] = true;
    } else {
      G2.active[i] = false;
    }
  }
  end_active = __rdtsc();
  cycles_active += end_active - start_active;

  run_graph_program(&deltastep, G2, 1, &ds_ts);
  deltastep.bid++;

  start_active = __rdtsc(); 
  //memset(G.active, 0, sizeof(bool)*(G.nvertices+1));
  bucket_not_empty = 0;
  //for (int i = 0 ; i < nthreads; i++)
  //  next_bucket[i*16] = INT_MAX;
  
  #pragma omp parallel for num_threads(nthreads) reduction(+:bucket_not_empty)
  for (int i = 0; i < G.nvertices; i++) {
    //int tid = omp_get_thread_num();
    if(G.vertexproperty[i].bucket == deltastep.bid) {
      G.active[i] = true;
    } else {
      G.active[i] = false;
    }
    if(G.vertexproperty[i].bucket >= deltastep.bid && G.vertexproperty[i].bucket < INT_MAX) {
      bucket_not_empty = 1;
      //next_bucket[tid*16] = std::min(next_bucket[tid*16], G.vertexproperty[i].bucket);
    }
  }
  end_active = __rdtsc();
  cycles_active += end_active - start_active;

  //printf("Vertex %d is in bucket %d \n", bucket_not_empty, G.vertexproperty[bucket_not_empty].bucket);
  
  //int next_non_empty_bucket = INT_MAX;
  //for (int i = 0; i < nthreads; i++) {
  //  next_non_empty_bucket = std::min(next_non_empty_bucket, next_bucket[i*16]);
  //}
  //printf("Next non-empty bucket = %d \n", next_non_empty_bucket);

  } while(bucket_not_empty);

  gettimeofday(&end, 0);
  printf("Time = %.3f ms \n", (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3);
  printf("Active setting time = %.3f ms \n", (cycles_active)/2.7e6);

  graph_program_clear(ds_ts);
  //graph_program_clear(lds_ts);
  //graph_program_clear(hds_ts);
 
  int reachable_vertices = 0;
  for (int i = 0; i < G.nvertices; i++) {
    if (G.vertexproperty[i].distance < MAX_DIST) {
      reachable_vertices++;
    }
  }
  printf("Reachable vertices = %d \n", reachable_vertices);

  for (int i = 0; i <= std::min(10, G.nvertices); i++) {
    G.vertexproperty[i].print();
    if (G.vertexproperty[i].distance < MAX_DIST) {
      printf("PATH: ");
      int par = i;
      while(par != -1) {
        printf(" %d ", par);
        par = G.vertexproperty[par].parent;
      }
      printf("\n");
    }
    //if (G.vertexproperty[i].depth < MAX_DIST) {
    //  printf("Depth %d : %d \n", i, G.vertexproperty[i].depth);
    //}
    //else {
    //  printf("Depth %d : INF \n", i);
    //}
  }

  /*FILE* f;
  f = fopen("out", "w");
  for (int i = 1; i <= G.nvertices; i++) {
      fprintf(f, "Depth %d : %d \n", i, G.vertexproperty[i].depth);
  }
  fclose(f);*/

}

int main (int argc, char* argv[]) {
  if (argc < 4) {
    printf("Correct format: %s A.light.mtx A.heavy.mtx delta\n", argv[0]);
    return 0;
  }

#ifdef __ASSERT
  printf("\nASSERT***************************************************Asserts are on.*************\n\n");
#endif

  //int NTHREADS = atoi(argv[3]);
  //int tid, nt;

#pragma omp parallel 
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
      printf("num threads got: %d\n", nthreads);
    }
  }
 
  int delta = atoi(argv[3]); 

  run_deltastepping(argv[1], argv[2], delta, nthreads);
  
}

