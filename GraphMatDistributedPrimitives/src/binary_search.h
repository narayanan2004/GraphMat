/******************************************************************************
 * ** Copyright (c) 2016, Intel Corporation                                     **
 * ** All rights reserved.                                                      **
 * **                                                                           **
 * ** Redistribution and use in source and binary forms, with or without        **
 * ** modification, are permitted provided that the following conditions        **
 * ** are met:                                                                  **
 * ** 1. Redistributions of source code must retain the above copyright         **
 * **    notice, this list of conditions and the following disclaimer.          **
 * ** 2. Redistributions in binary form must reproduce the above copyright      **
 * **    notice, this list of conditions and the following disclaimer in the    **
 * **    documentation and/or other materials provided with the distribution.   **
 * ** 3. Neither the name of the copyright holder nor the names of its          **
 * **    contributors may be used to endorse or promote products derived        **
 * **    from this software without specific prior written permission.          **
 * **                                                                           **
 * ** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
 * ** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
 * ** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
 * ** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
 * ** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
 * ** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
 * ** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
 * ** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
 * ** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
 * ** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
 * ** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * * ******************************************************************************/
/* Michael Anderson (Intel Corp.)
 *  * ******************************************************************************/


#ifndef SRC_BINARY_SEARCH_H_
#define SRC_BINARY_SEARCH_H_

template<typename T>
T binary_search_left_border(T * vec, T val,  long start,  long end,  long len)
{
  if(len == 0) return -1;

  // not found
  if(end < start)
  {
    return -1;
  }

  long mid = start + (end-start)/2;

  if(vec[mid] == val)
  {
    if(mid <= 0)
    {
      return mid;
    }
    else if(vec[mid-1] != vec[mid])
    {
      return mid;
    }
    else
    {
      return binary_search_left_border(vec, val, start, mid-1, len);
    }
  }
  else
  {
    if(val < vec[mid])
    {
      return binary_search_left_border(vec, val, start, mid-1, len);
    }
    else 
    {
      return binary_search_left_border(vec, val, mid+1, end, len);
    }
  }
}



template<typename T>
T binary_search_right_border(T * vec, T val,  long start,  long end,  long len)
{
  if(len == 0) return -1;

  // not found
  if(end < start)
  {
    return -1;
  }

  long mid = start + (end-start)/2;

  if(vec[mid] == val)
  {
    if(mid >= len-1)
    {
      return mid;
    }
    else if(vec[mid+1] != vec[mid])
    {
      return mid;
    }
    else
    {
      return binary_search_right_border(vec, val, mid+1, end, len);
    }
  }
  else
  {
    if(val < vec[mid])
    {
      return binary_search_right_border(vec, val, start, mid-1, len);
    }
    else 
    {
      return binary_search_right_border(vec, val, mid+1, end, len);
    }
  }
}

int l_binary_search(int start, int end, int * v, int item) {
  int e1 = start;
  int e2 = end;
  int eh;
  while(e2 >= e1)
  {
    eh = e2 - (e2 - e1) / 2;
    if(eh == 0) break;
    if(v[eh-1] < item && v[eh] >= item) break;
    if(v[eh] >= item)
    {
      e2 = eh-1;
    }
    else
    {
      e1 = eh+1;
    }
  }
  return eh;
}


int l_linear_search(int start, int end, int * v, int item) {
  if(v[0] >= item) return 0;
  for(int i = start ; i < end -1 ; i++)
  {
    if(v[i] < item && v[i+1] >= item) return i+1;
  }
  return end;
}


#endif  // SRC_BINARY_SEARCH_H_
