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
/* Narayanan Sundaram (Intel Corp.), Michael Anderson (Intel Corp.)
 * ******************************************************************************/

#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <sys/time.h>
#include <parallel/algorithm>
#include <omp.h>
#include <cassert>

namespace GraphMat {

inline double sec(struct timeval start, struct timeval end)
{
    return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))))/1.0e6;
}


template<class T>
void AddFn(T a, T b, T* c, void* vsp) {
  *c = a + b ;
}


template <class V, class E=int>
class Graph {

  public:
    int nvertices;
    long long int nnz;
    bool vertexpropertyowner;
    int tiles_per_dim;
    int num_threads;

    GraphMat::SpMat<GraphMat::DCSCTile<E> > *A;
    GraphMat::SpMat<GraphMat::DCSCTile<E> > *AT;
    GraphMat::SpVec<GraphMat::DenseSegment<V> > * vertexproperty;
    GraphMat::SpVec<GraphMat::DenseSegment<bool> > * active;
  
  public:
    Graph(): nvertices(0), nnz(0), vertexpropertyowner(true),
             tiles_per_dim(GraphMat::get_global_nrank()),
             A(nullptr), AT(nullptr), num_threads(omp_get_max_threads()),
             vertexproperty(nullptr), active(nullptr) {}
    void ReadEdgelist(GraphMat::edgelist_t<E> A_edges);
    void getVertexEdgelist(GraphMat::edgelist_t<V> & myedges);
    void getEdgelist(GraphMat::edgelist_t<E> & myedges);
    void ReadMTX(const char* filename); 
    void ReadGraphMatBin(const char* filename);
    void WriteGraphMatBin(const char* filename);

    void setAllActive();
    void setAllInactive();
    void setActive(int v);
    void setInactive(int v);
    void setAllVertexproperty(const V& val);
    void setVertexproperty(int v, const V& val);
    V getVertexproperty(int v) const;
    bool vertexNodeOwner(const int v) const;
    void saveVertexproperty(std::string fname, bool includeHeader=true) const;
    void reset();
    void shareVertexProperty(Graph<V,E>& g);
    int getNumberOfVertices() const;
    void applyToAllVertices(void (*ApplyFn)(V, V*, void*), void* param=nullptr);
    template<class T> void applyReduceAllVertices(T* val, void (*ApplyFn)(V*, T*, void*), void (*ReduceFn)(T,T,T*,void*)=AddFn<T>, void* param=nullptr);
    ~Graph();

  private:
    int vertexToNative(int vertex, int nsegments, int len) const;
    int nativeToVertex(int vertex, int nsegments, int len) const;

};



template<class V, class E>
int Graph<V,E>::vertexToNative(int vertex, int nsegments, int len) const
{
  if (true) {

    int v = vertex-1;
    int npartitions = num_threads * 16 * nsegments;
    int height = len / npartitions;
    int vmax = height * npartitions;
    if(v >= vmax)
    {
      return v+1;
    }
    int col = v%npartitions;
    int row = v/npartitions;
    return row + col * height+ 1;
  } else {
    return vertex;
  }
}

template<class V, class E>
int Graph<V,E>::nativeToVertex(int vertex, int nsegments, int len) const
{
  if (true) {
    int v = vertex-1;
    int npartitions = num_threads * 16 * nsegments;
    int height = len / npartitions;
    int vmax = height * npartitions;
    if(v >= vmax)
    {
      return v+1;
    }
    int col = v/height;
    int row = v%height;
    return col + row * npartitions+ 1;
  } else {
    return vertex;
  }
}

template<class V, class E>
void Graph<V,E>::ReadGraphMatBin(const char* filename) {
  std::stringstream fname_ss;
  fname_ss << filename << GraphMat::get_global_myrank();
  std::cout << "Reading file " << fname_ss.str() << std::endl;
  std::ifstream ifilestream(fname_ss.str().c_str(), std::ios::in|std::ios::binary);
  boost::archive::binary_iarchive bi(ifilestream);

  struct timeval start, end;
  gettimeofday(&start, 0);
  bi >> A;
  bi >> AT;
  tiles_per_dim = GraphMat::get_global_nrank();
  if(A->ntiles_x != tiles_per_dim || A->ntiles_y != tiles_per_dim || 
     AT->ntiles_x != tiles_per_dim || AT->ntiles_y != tiles_per_dim)   {
    std::cout << "Error reading file - mismatch in number of MPI ranks used in load vs save graph" << std::endl;
    exit(1);
  }

  bi >> num_threads;
  if(num_threads != omp_get_max_threads())   {
    std::cout << "Error reading file - mismatch in number of OpenMP threads used in load vs save graph" << std::endl;
    exit(1);
  } 

  nvertices = A->m;
  vertexproperty = new GraphMat::SpVec<GraphMat::DenseSegment<V> >(A->m, tiles_per_dim, GraphMat::vector_partition_fn);
  V *__v = new V;
  vertexproperty->setAll(*__v);
  delete __v;

  active = new GraphMat::SpVec<GraphMat::DenseSegment<bool> >(A->m, tiles_per_dim, GraphMat::vector_partition_fn);
  active->setAll(false);

  vertexpropertyowner = true;
  nnz = A->getNNZ();
  
  gettimeofday(&end, 0);
  std::cout << "Finished GraphMat read + construction, time: " << sec(start,end)  << std::endl;

  ifilestream.close();
  MPI_Barrier(MPI_COMM_WORLD);
}

template<class V, class E>
void Graph<V,E>::WriteGraphMatBin(const char* filename) {
  std::stringstream fname_ss;
  fname_ss << filename << GraphMat::get_global_myrank();
  std::cout << "Writing file " << fname_ss.str() << std::endl;
  std::ofstream ofilestream(fname_ss.str().c_str(), std::ios::out|std::ios::binary);
  boost::archive::binary_oarchive bo(ofilestream);
  bo << A;
  bo << AT;
  bo << num_threads;
  ofilestream.close();
  MPI_Barrier(MPI_COMM_WORLD);
}

template<class V, class E>
void Graph<V,E>::ReadEdgelist(GraphMat::edgelist_t<E> A_edges) {

  struct timeval start, end;
  gettimeofday(&start, 0);
  
  tiles_per_dim = GraphMat::get_global_nrank();
  num_threads = omp_get_max_threads();
    
  #pragma omp parallel for
  for(int i = 0 ; i < A_edges.nnz ; i++)
  {
    A_edges.edges[i].src = vertexToNative(A_edges.edges[i].src, tiles_per_dim, A_edges.m);
    A_edges.edges[i].dst = vertexToNative(A_edges.edges[i].dst, tiles_per_dim, A_edges.m);
  }

  A = new GraphMat::SpMat<GraphMat::DCSCTile<E> >(A_edges, tiles_per_dim, tiles_per_dim, GraphMat::partition_fn_2d);
  GraphMat::Transpose(A, &AT, tiles_per_dim, tiles_per_dim, GraphMat::partition_fn_2d);

  int m_ = A->m;
  assert(A->m == A->n);
  nnz = A->getNNZ();
  vertexproperty = new GraphMat::SpVec<GraphMat::DenseSegment<V> >(A->m, tiles_per_dim, GraphMat::vector_partition_fn);
  V *__v = new V;
  vertexproperty->setAll(*__v);
  delete __v;
  active = new GraphMat::SpVec<GraphMat::DenseSegment<bool> >(A->m, tiles_per_dim, GraphMat::vector_partition_fn);
  active->setAll(false);

  nvertices = m_;
  vertexpropertyowner = true;
  
  gettimeofday(&end, 0);
  std::cout << "Finished GraphMat read + construction, time: " << sec(start,end)  << std::endl;


}

template<class V, class E>
void Graph<V,E>::ReadMTX(const char* filename) {
  GraphMat::edgelist_t<E> A_edges;
  GraphMat::load_edgelist(filename, &A_edges, true, true, true);// binary format with header and edge weights

  if (A_edges.m != A_edges.n) {
    auto maxn = std::max(A_edges.m, A_edges.n);
    A_edges.m = maxn;
    A_edges.n = maxn;
  }
  ReadEdgelist(A_edges);
  A_edges.clear();
}


template<class V, class E> 
void Graph<V,E>::setAllActive() {
  active->setAll(true);
}

template<class V, class E> 
void Graph<V,E>::setAllInactive() {
  active->setAll(false);
  int global_myrank = GraphMat::get_global_myrank();
  for(int segmentId = 0 ; segmentId < active->nsegments ; segmentId++)
  {
    if(active->nodeIds[segmentId] == global_myrank)
    {
      GraphMat::DenseSegment<bool>* s1 = active->segments[segmentId];
      GraphMat::clear_dense_segment(s1->properties->value, s1->properties->bit_vector, s1->num_ints);
    }
  }
}

template<class V, class E> 
void Graph<V,E>::setActive(int v) {
  int v_new = vertexToNative(v, tiles_per_dim, nvertices);
  active->set(v_new, true);
}

template<class V, class E> 
void Graph<V,E>::setInactive(int v) {
  int v_new = vertexToNative(v, tiles_per_dim, nvertices);
  active->unset(v_new);
}
template<class V, class E> 
void Graph<V,E>::reset() {
  setAllInactive();
  V v;
  vertexproperty->setAll(v);
}

template<class V, class E> 
void Graph<V,E>::shareVertexProperty(Graph<V,E>& g) {
  if (vertexproperty != nullptr) delete vertexproperty;
  vertexproperty = g.vertexproperty;
  vertexpropertyowner = false;
}

template<class V, class E> 
void Graph<V,E>::setAllVertexproperty(const V& val) {
  vertexproperty->setAll(val);
}

template<class V, class E> 
void Graph<V,E>::setVertexproperty(int v, const V& val) {
  int v_new = vertexToNative(v, tiles_per_dim, nvertices);
  vertexproperty->set(v_new, val);
}

template<class V, class E> 
void Graph<V,E>::getVertexEdgelist(GraphMat::edgelist_t<V> & myedges) {
  vertexproperty->get_edges(&myedges);
  for(unsigned int i = 0 ; i < myedges.nnz ; i++)
  {
    myedges.edges[i].src = nativeToVertex(myedges.edges[i].src, tiles_per_dim, nvertices);
  }
}

template<class V, class E> 
void Graph<V,E>::getEdgelist(GraphMat::edgelist_t<E> & myedges) {
  A->get_edges(&myedges);
  for(unsigned int i = 0 ; i < myedges.nnz ; i++)
  {
    myedges.edges[i].src = nativeToVertex(myedges.edges[i].src, tiles_per_dim, nvertices);
  }
}

template<class V, class E> 
void Graph<V,E>::saveVertexproperty(std::string fname, bool includeHeader) const {
  GraphMat::edgelist_t<V> myedges;
  vertexproperty->get_edges(&myedges);
  for(unsigned int i = 0 ; i < myedges.nnz ; i++)
  {
    myedges.edges[i].src = nativeToVertex(myedges.edges[i].src, tiles_per_dim, nvertices);
  }
  GraphMat::SpVec<GraphMat::DenseSegment<V> > * vertexproperty2 = new GraphMat::SpVec<GraphMat::DenseSegment<V> >(nvertices, tiles_per_dim, GraphMat::vector_partition_fn);
  vertexproperty2->ingestEdgelist(myedges);
  myedges.clear();
  vertexproperty2->save(fname, includeHeader);
  delete vertexproperty2;
}

template<class V, class E>
bool Graph<V,E>::vertexNodeOwner(const int v) const {
  int v_new = vertexToNative(v, tiles_per_dim, nvertices);
  return vertexproperty->node_owner(v_new);
}

template<class V, class E> 
V Graph<V,E>::getVertexproperty(const int v) const {
  V vp ;
  int v_new = vertexToNative(v, tiles_per_dim, nvertices);
  vertexproperty->get(v_new, &vp);
  return vp;
}

template<class V, class E> 
int Graph<V,E>::getNumberOfVertices() const {
  return nvertices;
}

template<class V, class E> 
void Graph<V,E>::applyToAllVertices( void (*ApplyFn)(V, V*, void*), void* param) {
  GraphMat::Apply(vertexproperty, vertexproperty, ApplyFn, param);
}


template<class V, class E> 
template<class T> 
void Graph<V,E>::applyReduceAllVertices(T* val, void (*ApplyFn)(V*, T*, void*), void (*ReduceFn)(T,T,T*,void*), void* param) {
  GraphMat::MapReduce(vertexproperty, val, ApplyFn, ReduceFn, param);
}

template<class V, class E> 
Graph<V,E>::~Graph() {
  if (A != nullptr) {
	delete A;
	A = nullptr;
  }
  if (AT != nullptr) {
	delete AT;
	AT = nullptr;
  }
  if (vertexpropertyowner) {
    if (vertexproperty != nullptr) {
  	delete vertexproperty;
	vertexproperty = nullptr;
    }
  }
  if (active != nullptr) {
	delete active;
	active = nullptr;
  }
}

} //namespace GraphMat
