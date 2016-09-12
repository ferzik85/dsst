#pragma once

#ifndef _SSE_HPP_
#define _SSE_HPP_

#include <emmintrin.h> // SSE2

// set, load and store values
inline __m128 SET( const float &x ) { return _mm_set_ps1(x); }
inline __m128 SET( float x, float y, float z, float w ) { return _mm_set_ps(x,y,z,w); }
inline __m128i SET( const int &x ) { return _mm_set1_epi32(x); }
inline __m128 LD( const float &x ) { return _mm_load_ps(&x); }
inline __m128 LDu( const float &x ) { return _mm_loadu_ps(&x); }
inline __m128 STR( float &x, const __m128 y ) { _mm_store_ps(&x,y); return y; }
inline __m128 STR1( float &x, const __m128 y ) { _mm_store_ss(&x,y); return y; }
inline __m128 STRu( float &x, const __m128 y ) { _mm_storeu_ps(&x,y); return y; }
inline __m128 STR( float &x, const float y ) { return STR(x,SET(y)); }

// arithmetic operators
inline __m128i ADD( const __m128i x, const __m128i y ) { return _mm_add_epi32(x,y); }
inline __m128 ADD( const __m128 x, const __m128 y ) { return _mm_add_ps(x,y); }
inline __m128 ADD( const __m128 x, const __m128 y, const __m128 z ) {return ADD(ADD(x,y),z); }
inline __m128 ADD( const __m128 a, const __m128 b, const __m128 c, const __m128 &d ) {return ADD(ADD(ADD(a,b),c),d); }
inline __m128 SUB( const __m128 x, const __m128 y ) { return _mm_sub_ps(x,y); }
inline __m128 MUL( const __m128 x, const __m128 y ) { return _mm_mul_ps(x,y); }
inline __m128 MUL( const __m128 x, const float y ) { return MUL(x,SET(y)); }
inline __m128 MUL( const float x, const __m128 y ) { return MUL(SET(x),y); }
inline __m128 INC( __m128 &x, const __m128 y ) { return x = ADD(x,y); }
inline __m128 INC( float &x, const __m128 y ) { __m128 t=ADD(LD(x),y); return STR(x,t); }
inline __m128 DEC( __m128 &x, const __m128 y ) { return x = SUB(x,y); }
inline __m128 DEC( float &x, const __m128 y ) { __m128 t=SUB(LD(x),y); return STR(x,t); }
inline __m128 MIN( const __m128 x, const __m128 y ) { return _mm_min_ps(x,y); }
inline __m128 RCP( const __m128 x ) { return _mm_rcp_ps(x); }
inline __m128 RCPSQRT( const __m128 x ) { return _mm_rsqrt_ps(x); }

// logical operators
inline __m128 AND( const __m128 x, const __m128 y ) { return _mm_and_ps(x,y); }
inline __m128i AND( const __m128i x, const __m128i y ) { return _mm_and_si128(x,y); }
inline __m128 ANDNOT( const __m128 x, const __m128 y ) { return _mm_andnot_ps(x,y); }
inline __m128 OR( const __m128 x, const __m128 y ) { return _mm_or_ps(x,y); }
inline __m128 XOR( const __m128 x, const __m128 y ) { return _mm_xor_ps(x,y); }

// comparison operators
inline __m128 CMPGT( const __m128 x, const __m128 y ) { return _mm_cmpgt_ps(x,y); }
inline __m128 CMPLT( const __m128 x, const __m128 y ) { return _mm_cmplt_ps(x,y); }
inline __m128i CMPGT( const __m128i x, const __m128i y ) { return _mm_cmpgt_epi32(x,y); }
inline __m128i CMPLT( const __m128i x, const __m128i y ) { return _mm_cmplt_epi32(x,y); }

// conversion operators
inline __m128 CVT( const __m128i x ) { return _mm_cvtepi32_ps(x);  }
inline __m128i CVT( const __m128 x  ) { return _mm_cvttps_epi32(x); }

#endif
