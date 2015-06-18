#ifndef __UTILS_H
#define __UTILS_H


#include "immintrin.h"

//SSE 
#define CPU_FREQ (2.7e9)

#define SIMD_WIDTH (4)
#define SIMDINTTYPE         __m128i
#define SIMDMASKTYPE         __m128i
#define _MM_LOADU(Addr)     _mm_loadu_si128((__m128i *)(Addr))
#define _MM_CMP_EQ(A,B)     _mm_cmpeq_epi32(A,B)
#define _MM_SET1(val)           _mm_set1_epi32(val)


#endif
