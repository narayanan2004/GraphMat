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
#include <omp.h>
#include "GraphMatRuntime.cpp"
#include <algorithm>
#include <iomanip>

const int MAX_THREADS = 128;

template <unsigned int K>
class LatentVector {
  public:
    double N[K];
    char type;

  public:
    LatentVector() {
    }
    ~LatentVector() {
    }

    bool operator !=(const LatentVector<K>& p) {
      bool result = false;
      for (int i = 0; i < K; i++) {
        if (fabs(p.N[i] - N[i]) > 1e-3) {
          result = true;
        }
      }
      return result;
    }

    void print() const {
      printf("%c ", type);
      for (int i = 0; i < K; i++) {
        printf(" %.2f ", N[i]);
      }
    }

};
template<unsigned int K>
class LDAInitProgram : public GraphProgram<LatentVector<K>, LatentVector<K>, LatentVector<K> > {

  public:
    LDAInitProgram() {
      this->order = ALL_EDGES;// check
      this->activity = ALL_VERTICES;
    }


  void reduce_function(LatentVector<K>& v, const LatentVector<K>& w) const {
    for (int i = 0; i < K; i++) v.N[i] += w.N[i];
  }

  void process_message(const LatentVector<K>& message, const int edge_value, 
                        const LatentVector<K>& vertexprop, LatentVector<K>& res) const {
    double gamma_wjk[K];

    double sum = 0;

    //Introduce randomness, but ensure that the random number is 
    //consistent for both directions of the edge.
    unsigned int rstart = edge_value; 
    for (int i = 0; i < K; i++) {
      gamma_wjk[i] = (double)rand_r(&rstart)/RAND_MAX; //double(i+1)/K;
      sum += gamma_wjk[i];
    }

    for (int i = 0; i < K; i++) {
      res.N[i] = gamma_wjk[i]/sum*(double)edge_value;
    }
  }
  bool send_message(const LatentVector<K>& vertexprop, LatentVector<K>& message) const {
    message = vertexprop;
    return true;
  }

  void apply(const LatentVector<K>& message_out, LatentVector<K>& vertexprop) {
    for (int i = 0; i < K; i++) {
      vertexprop.N[i] = message_out.N[i];
    }
  }
};

template<unsigned int K>
void IfTerm(LatentVector<K>* v, LatentVector<K>* out, void* param) {
  if (v->type == 'w') {
    for (int i = 0; i < K; i++) out->N[i] = v->N[i];
  } else {
    for (int i = 0; i < K; i++) out->N[i] = 0;
  }
}

template<unsigned int K>
void Add(LatentVector<K> v1, LatentVector<K> v2, LatentVector<K>* out, void* param) {
    for (int i = 0; i < K; i++) out->N[i] = v1.N[i] + v2.N[i];
}

template<unsigned int K>
class LDAProgram : public GraphProgram<LatentVector<K>, LatentVector<K>, LatentVector<K> > {
  public:
    double alpha;
    double eta;
    double vocab_size;
    LatentVector<K> global_N;
    Graph<LatentVector<K> >& graph_ref;

  public:
    LDAProgram(Graph<LatentVector<K> >& G, double a, double e, double V) : 
                graph_ref(G), alpha(a), eta(e), vocab_size(V) {
      this->order = ALL_EDGES;// check
      this->activity = ALL_VERTICES;
    }

    void calcGlobalN() {
      for (int i = 0; i < K; i++) global_N.N[i] = 0;
      graph_ref.applyReduceAllVertices(&global_N, IfTerm, Add);
      //std::cout << "global_N " <<std::endl;
      //for (int i = 0; i < K; i++) std::cout << global_N.N[i] << " ";
      //std::cout << std::endl;
    }

  void reduce_function(LatentVector<K>& v, const LatentVector<K>& w) const {
    for (int i = 0; i < K; i++) v.N[i] += w.N[i];
  }

  void process_message(const LatentVector<K>& message, const int edge_value, 
                        const LatentVector<K>& vertexprop, LatentVector<K>& res) const {
    double gamma_wjk[K];
    double my_offset, other_offset;

    if (vertexprop.type == 'd') {
      my_offset = alpha; 
      other_offset = eta;
    } else {
      my_offset = eta; 
      other_offset = alpha;
    }

    double sum = 0;
    for (int i = 0; i < K; i++) {
      gamma_wjk[i] = (vertexprop.N[i] + my_offset -1.0)*(message.N[i] + other_offset - 1.0)/
                    (global_N.N[i] + vocab_size*(eta - 1.0));
      sum += gamma_wjk[i];
    }

    for (int i = 0; i < K; i++) {
      res.N[i] = gamma_wjk[i]/sum*(double)edge_value;
    }
  }

  bool send_message(const LatentVector<K>& vertexprop, LatentVector<K>& message) const {
    message = vertexprop;
    return true;
  }

  void apply(const LatentVector<K>& message_out, LatentVector<K>& vertexprop) {
    for (int i = 0; i < K; i++) {
      vertexprop.N[i] = message_out.N[i];
    }
  }


  void do_every_iteration(int iteration_number) {
    calcGlobalN();
  }
};

int getnthelement(const double *arr_in, int n, int p) {
  double* arr = new double[n];
  memcpy(arr, arr_in, sizeof(double)*n);
  std::sort(arr, arr+n);
  auto pos = std::search_n(arr_in, arr_in+n, 1, arr[p]);
  return std::distance(arr_in, pos);
}

void run_lda(char* filename, int nthreads, int ndoc, int nterms, int niterations=10) {
  const int k = 20;
  Graph< LatentVector<k> > G;
  G.ReadMTX(filename, nthreads*8); //nthread pieces of matrix
  if (ndoc + nterms != G.getNumberOfVertices()) {
    std::cout << "Number of vertices in graph != NDOC + NTERMS" << std::endl;
    exit(1);
  }

  for (int i = 1; i <= G.getNumberOfVertices(); i++) {
    LatentVector<k> v;
    if ( i <= ndoc) {
      v.type = 'd';
    } else {
      v.type = 'w';
    }
    G.setVertexproperty(i, v);
  }

//  initializeWeights(G);
  LDAInitProgram<k> ldainit_program;
  G.setAllActive();
  run_graph_program(&ldainit_program, G, 1);
  /*for (int i = 1; i <= std::min(10, G.getNumberOfVertices()); i++) { 
    printf("%d : ", i) ;
    G.getVertexproperty(i).print();
    printf("\n");
  }*/


  LDAProgram<k> ldap(G, 1, 1, nterms);
  ldap.calcGlobalN();
  auto ldap_tmp = graph_program_init(ldap, G);

  printf("LDA Init over\n");
  
  struct timeval start, end;

  gettimeofday(&start, 0);

  G.setAllActive();
  run_graph_program(&ldap, G, niterations, &ldap_tmp);

  gettimeofday(&end, 0);
  
  double time = (end.tv_sec-start.tv_sec)*1e3+(end.tv_usec-start.tv_usec)*1e-3;
  printf("Time = %.3f ms \n", time);

  graph_program_clear(ldap_tmp);

  for (int i = 1; i <= std::min(5, ndoc); i++) { 
    printf("%d : ", i) ;
    G.getVertexproperty(i).print();
    printf("\n");
  }
  for (int i = 1; i <= std::min(5, nterms); i++) { 
    printf("%d : ", i+ndoc) ;
    G.getVertexproperty(i+ndoc).print();
    printf("\n");
  }

  //Calculate topic-word distributions
  double *phi = new double[k*nterms];
  for (int i = 0; i < nterms; i++){
    assert(G.getVertexproperty(i+ndoc+1).type == 'w');
    for (int j = 0; j < k; j++) { 
      phi[i + j*nterms] = G.getVertexproperty(i + ndoc + 1).N[j]; 
    }
  }
  for (int j = 0; j < k; j++) {
    double sum = 0;
    double* phi_j = phi + j*nterms;
    for (int i = 0; i < nterms; i++) {
      sum += phi_j[i];
    }
    for (int i = 0; i < nterms; i++) {
      phi_j[i] /= sum;
    }
  }

  //Print top 5 words in each topic
  std::cout << "Top 5 words in each topic" << std::endl;
  for (int j = 0; j < k; j++) {
    double* phi_j = phi + j*nterms;
    std::cout << "Topic " << j << std::endl;
    for (int i = 0; i < std::min(5, nterms); i++) {
      std::cout << "Word # ";
      int pos = getnthelement(phi_j, nterms, nterms-i-1);
      std::cout << pos + ndoc + 1 << " " << std::setprecision(3) << phi_j[pos] << std::endl;
    }
  }

  delete [] phi;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    printf("Correct format: %s A.mtx #DOC #TERMS {#iterations (default 10)}\n", argv[0]);
    return 0;
  }
  int ndoc = atoi(argv[2]);
  int nterms = atoi(argv[3]);

  int niterations = (argc >= 5)?(atoi(argv[4])):(10);


#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
      printf("num threads got: %d\n", nthreads);
    }
  }
  

  run_lda(argv[1], nthreads, ndoc, nterms, niterations); 
  
}

