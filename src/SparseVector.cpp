#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <immintrin.h>
#include "utils.h"
#include <iostream>

inline void set_bitvector(unsigned int idx, int* bitvec) {
    unsigned int neighbor_id = idx;
    int dword = (neighbor_id >> 5);
    int bit = neighbor_id  & 0x1F;
    unsigned int current_value = bitvec[dword];
    if ( (current_value & (1<<bit)) == 0)
    {
      bitvec[dword] = current_value | (1<<bit);
    }
}

inline bool get_bitvector(unsigned int idx, const int* bitvec) {
    unsigned int neighbor_id = idx;
    int dword = (neighbor_id >> 5);
    int bit = neighbor_id  & 0x1F;
    unsigned int current_value = bitvec[dword];
    return ( (current_value & (1<<bit)) );
}

//------------------------------------------------------------

template <class T>
class SparseVector {
  public:

  int *bitvector;
  T* value;
  int length;
  int numInts;

  public:

  SparseVector(int n);
  SparseVector();
  void set(int idx, const T& v);
  const T& getValue(int i) const ;
  void resize(const int n);
  void clear();
  template<class U, class V>
  void reduce(const int idx, const T& v, const GraphProgram<U, T, V>* gp );
  void print();
  int nnz() const;
  ~SparseVector();
};

template<class T> using SparseInVector = SparseVector<T>;
template<class T> using SparseOutVector = SparseVector<T>;

//------------------------------------------------------------

template <class T>
void SparseVector<T>::set(int idx, const T& v) {
    set_bitvector(idx, bitvector);
    value[idx] = v;
}

template <class T>
const T& SparseVector<T>::getValue(int i) const {
  #ifdef __ASSERT
    assert( i>=0 && i < length);
  #endif
    return value[i];
}

template <class T>
SparseVector<T>::SparseVector() {
  length = 0;
  numInts = 0;
  value = NULL;
  bitvector = NULL;
}

template <class T>
SparseVector<T>::SparseVector(int n) {
  length = n;
  value = new T[length];
  numInts = std::max(SIMD_WIDTH, ((length/32+SIMD_WIDTH)/SIMD_WIDTH)*SIMD_WIDTH); //multiple of SIMD_WIDTH
  bitvector = new int[numInts];
  memset(bitvector, 0, (numInts)*sizeof(int));
}

template <class T>
SparseVector<T>::~SparseVector() {
  length = 0;
  if (value) {
    delete [] value;
    value = NULL;
  }
  if (bitvector) {
    delete [] bitvector;
    bitvector = NULL;
  }
}

template <class T>
void SparseVector<T>::clear() {
  if (length) {
    memset(bitvector, 0, (numInts)*sizeof(int));
  }
}

template <class T>
void SparseVector<T>::resize(int n) {
  length = 0;
  if (value) {
    delete [] value;
    value = NULL;
  }
  if (bitvector) {
    delete [] bitvector;
    bitvector = NULL;
  }
  length = n+1;
  value = new T[length];
  numInts = std::max(SIMD_WIDTH, ((length/32+SIMD_WIDTH)/SIMD_WIDTH)*SIMD_WIDTH); //multiple of SIMD_WIDTH
  bitvector = new int[numInts];
  memset(bitvector, 0, (numInts)*sizeof(int));
}

template <class T>
void SparseVector<T>::print() {
  std::cout << "Printing vector \n";
  for (int i = 0; i <= length; i++) {
    if (get_bitvector(i, bitvector)) {
      std::cout << i << " " << value[i] << "\n";
    }
  }
}

template <class T>
int SparseVector<T>::nnz() const {
    int len = 0;
    #pragma omp parallel for num_threads(nthreads) schedule(guided, 128) reduction(+:len)
    for (int ii = 0; ii < numInts; ii++) {
      int p = _popcnt32(bitvector[ii]);
      len += p;
    }
    return len;
}

template <class T>
template <class U, class V>
void SparseVector<T>::reduce(int idx, const T& v, const GraphProgram<U, T, V>* gp ) {

  if (get_bitvector(idx, bitvector)) {
    gp->reduce_function(value[idx], v);
  } else {
    set_bitvector(idx, bitvector);
    value[idx] = v;
  }
}

//-------------------------------------
