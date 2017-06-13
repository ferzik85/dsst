#include "gradient.h"
#include "string.h"
#include <math.h>
#include <malloc.h>
#include <fftw3.h>

#ifdef SSEv2 
  #include "sse.hpp"
#endif

#define PI 3.14159265f

// compute x and y gradients for just one column 
void grad1( float *I, float *Gx, float *Gy, int h, int w, int x ) { 
  int y, y1; float *Ip, *In, r; 
#ifdef SSEv2
  __m128 *_Ip, *_In, *_G, _r;
#endif
  // compute column of Gx
  Ip=I-h; In=I+h; r=.5f;
  if(x==0) { r=1; Ip+=h; } else if(x==w-1) { r=1; In-=h; }
  if( h<4 || h%4>0 || (size_t(I)&15) || (size_t(Gx)&15) ) {
    for( y=0; y<h; y++ ) *Gx++=(*In++-*Ip++)*r;
  } else {  
#ifdef SSEv2
    _G=(__m128*) Gx; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = SET(r);
	for(y=0; y<h; y+=4) *_G++=MUL(SUB(*_In++,*_Ip++),_r);
#else
    for(y=0; y<h; y++) Gx[y]=(In[y]-Ip[y])*r;
#endif
  }

  // compute column of Gy b
  #define GRADY(r) *Gy++=(*In++-*Ip++)*r;
  Ip=I; In=Ip+1;

  // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
 /* y1=((~((size_t) Gy) + 1) & 15)/4;
  if(y1==0) y1=4; 
  if(y1>h-1) y1=h-1;
  GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);*/

#ifdef SSEv2
   y1=((~((size_t) Gy) + 1) & 15)/4;
  if(y1==0) y1=4; 
  if(y1>h-1) y1=h-1;
  GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);

  _r = SET(.5f); _G=(__m128*) Gy;
  for(; y+4<h-1; y+=4, Ip+=4, In+=4, Gy+=4)
	  *_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
#else
  y1=h-1;
  GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);
  for(; y+4<h-1; y++) 
	  Gy[y]=(In[y]-Ip[y])*0.5f;
#endif

  for(; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

// compute x and y gradients at each location (uses sse)
void grad2(float *I, float *Gx, float *Gy, int h, int w, int d) {
	int o, x, c, a = w*h; for (c = 0; c<d; c++) for (x = 0; x<w; x++) {
		o = c*a + x*h; grad1(I + o, Gx + o, Gy + o, h, w, x);
	}
}

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float* acosTable() {
  const int n=10000, b=10; int i;
  static float a[n*2+b*2];
  static bool init=false;
  float *a1=a+n+b; if( init ) return a1;
  for( i=-n-b; i<-n; i++ )   a1[i]=PI;
  for( i=-n; i<n; i++ )      a1[i]=float(acos(i/float(n)));
  for( i=n; i<n+b; i++ )     a1[i]=0;
  for( i=-n-b; i<n/10; i++ ) if( a1[i] > PI-1e-6f ) a1[i]=PI-1e-6f;
  init=true; return a1;
}

void gradientMagnitude( float *I, float *M, float *O, int h, int w, int d, bool full ) {
  int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; 
  float *acost = acosTable(), acMult=10000.0f;
  // allocate memory for storing one column of output (padded so h4%4==0)
  h4=(h%4==0) ? h : h-(h%4)+4; s=d*h4*sizeof(float);

#ifdef SSEv2
   __m128 *_Gx, *_Gy, *_M2, _m;
  M2=(float*) _aligned_malloc(s,16); 
  Gx=(float*) _aligned_malloc(s,16); 
  Gy=(float*) _aligned_malloc(s,16); 
  memset(M2,0,s);
  memset(Gx,0,s);
  memset(Gy,0,s);
  _M2=(__m128*) M2; _Gx=(__m128*) Gx; _Gy=(__m128*) Gy;
#else
   M2 = (float*)fftwf_malloc(sizeof(float)*d*h4);
   Gx = (float*)fftwf_malloc(sizeof(float)*d*h4);
   Gy = (float*)fftwf_malloc(sizeof(float)*d*h4);
   memset(M2,0,s);
   memset(Gx,0,s);
   memset(Gy,0,s);
#endif

  // compute gradient magnitude and orientation for each column
  for( x=0; x<w; x++ ) {
    // compute gradients (Gx, Gy) with maximum squared magnitude (M2)	
    for(c=0; c<d; c++) {
      grad1( I+x*h+c*w*h, Gx+c*h4, Gy+c*h4, h, w, x );
#ifdef SSEv2
      for( y=0; y<h4/4; y++ ) { 
        y1=(h4/4)*c+y;
        _M2[y1]=ADD(MUL(_Gx[y1],_Gx[y1]),MUL(_Gy[y1],_Gy[y1]));
        if( c==0 ) continue; 
		_m = CMPGT( _M2[y1], _M2[y] );
        _M2[y] = OR( AND(_m,_M2[y1]), ANDNOT(_m,_M2[y]) );
        _Gx[y] = OR( AND(_m,_Gx[y1]), ANDNOT(_m,_Gx[y]) );
        _Gy[y] = OR( AND(_m,_Gy[y1]), ANDNOT(_m,_Gy[y]) );
      }

#else
	  for( y=0; y<h4; y++ ) {
        y1=h4*c+y;	
        M2[y1]=(Gx[y1]*Gx[y1]+Gy[y1]*Gy[y1]);

        if( c==0 ) continue; 
			M2[y] = M2[y1] > M2[y] ? M2[y1] : M2[y];
			Gx[y] = M2[y1] > M2[y] ? Gx[y1] : Gx[y];
			Gy[y] = M2[y1] > M2[y] ? Gy[y1] : Gy[y];
      }
#endif
	  
    }
    // compute gradient mangitude (M) and normalize Gx
#ifdef SSEv2
	  for( y=0; y<h4/4; y++ ) {
      _m = MIN( RCPSQRT(_M2[y]), SET(1e10f) );
      _M2[y] = RCP(_m); 
      if(O) _Gx[y] = MUL( MUL(_Gx[y],_m), SET(acMult) );
      if(O) _Gx[y] = XOR( _Gx[y], AND(_Gy[y], SET(-0.f)) );
    };

#else
	  for( y=0; y<h4; y++ ) { 

	   //----небольшие отличия в ссе версии из-за этого участка кода
	   float m =  1.0f/sqrt((float)M2[y]);
	   m = m < 1e10f ? m : 1e10f;  
       M2[y] = 1.0f/m; 
	   //----------------

		/*float m = 1.0f/sqrt((float)M2[y]);
	    if (m < 1e10f)
			M2[y] = (float)sqrt((float)M2[y]);
		else {
			M2[y] = 1e-10f;
			m = 1e10f;
		}
       */

	   if(O) Gx[y] = (Gx[y]*m)*acMult;
	 
	   if(O) {
		   //bitwise AND on floats
		   float zero = -0.f; 
		   unsigned char *pGy = 0;
		   unsigned char *pZero = 0;
		   unsigned char *pGand = 0;
		   float Gand = 0.f;	  
		   pGy   = reinterpret_cast<unsigned char *>(&Gy[y]);
		   pZero = reinterpret_cast<unsigned char *>(&zero);
		   pGand = reinterpret_cast<unsigned char *>(&Gand);
		   for (int i=0; i<4; i++){
			   *pGand = (*pGy & *pZero);
			   pGand++;
			   pGy++;
			   pZero++;
		   };

		   //bitwise XOR on floats
		   unsigned char *pGx = 0;
		   unsigned char *pGxor = 0;
		   float Gxor = 0;
		   pGx   = reinterpret_cast<unsigned char *>(&Gx[y]);
		   pGand = reinterpret_cast<unsigned char *>(&Gand);
		   pGxor = reinterpret_cast<unsigned char *>(&Gxor);
		   for (int i=0; i<4; i++){
			   *pGxor = (*pGx ^ *pGand);
			   pGxor++;
			   pGx++;
			   pGand++;
		   };

		   Gx[y] = Gxor;
	   };
    };
#endif

    memcpy( M+x*h, M2, h*sizeof(float) );
    // compute and store gradient orientation (O) via table lookup
    if( O!=0 ) for( y=0; y<h; y++ ) 
		O[x*h+y] = acost[(int)Gx[y]];
    if( O!=0 && full ) {

      y1=((~size_t(O+x*h)+1)&15)/4; y=0;
      for( ; y<y1; y++ ) O[y+x*h]+=(Gy[y]<0)*PI; 

#ifdef SSEv2
      for( ; y<h-4; y+=4 ) STRu( O[y+x*h],
        ADD( LDu(O[y+x*h]), AND(CMPLT(LDu(Gy[y]),SET(0.f)),SET(PI)) ) );
#else
	  for( ; y<h-4; y++ )  
		  O[y+x*h]+=(Gy[y]<0)*PI;
		 // if (Gy[y] < 0) {
		 //	  O[y+x*h] += PI;
		 // }
		  //else O[y+x*h]++;
		  //else O[y+x*h]++;
#endif
      for( ; y<h; y++) O[y+x*h]+=(Gy[y]<0)*PI;
    }
  }

#ifdef SSEv2
 _aligned_free(M2); 
 _aligned_free(Gx); 
 _aligned_free(Gy); 
#else
  fftwf_free(M2);
  fftwf_free(Gx);
  fftwf_free(Gy);
#endif
}


// normalize gradient magnitude at each location (uses sse)
void gradMagNormalization( float *M, float *S, int h, int w, float norm ) {

#ifdef SSEv2
	__m128 *_M, *_S, _norm; 
	_S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
	bool sse = !(size_t(M)&15) && !(size_t(S)&15);
#endif

	int i=0, n=h*w;
	int n4=n/4;

#ifdef SSEv2
	if(sse) { 
		for(; i<n4; i++)   
			*_M++=MUL(*_M,RCP(ADD(*_S++,_norm))); i*=4;
	}
#else
	for(; i<n; i++) M[i] /= (S[i] + norm); //i*=4; 
#endif

	for(; i<n; i++) M[i] /= (S[i] + norm);
}

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 
void gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
				  int nb, int n, float norm, int nOrients, bool full, bool interpolate )
{
	// assumes all *OUTPUT* matrices are 4-byte aligned
	int i, o0, o1; float o, od, m;
	const float oMult=(float)nOrients/(full?2*PI:PI); const int oMax=nOrients*nb;
#ifdef SSEv2
	__m128i _o0, _o1, *_O0, *_O1; 
	__m128 _o, _od, _m, *_M0, *_M1;
	const __m128 _norm=SET(norm), _oMult=SET(oMult), _nbf=SET((float)nb);
	const __m128i _oMax=SET(oMax), _nb=SET(nb);
	_O0=(__m128i*) O0; 
	_O1=(__m128i*) O1; 
	_M0=(__m128*) M0; 
	_M1=(__m128*) M1;
#endif

 	if( interpolate ) 
#ifdef SSEv2
		for( i=0; i<=n-4; i+=4 ) {
			_o=MUL(LDu(O[i]),_oMult); _o0=CVT(_o); _od=SUB(_o,CVT(_o0));
			_o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
			_o1=ADD(_o0,_nb); _o1=AND(CMPGT(_oMax,_o1),_o1); *_O1++=_o1;
			_m=MUL(LDu(M[i]),_norm); *_M1=MUL(_od,_m); *_M0++=SUB(_m,*_M1); _M1++;
		} 
#else
		for( i=0; i<=n-4; i++ ) {
			o = O[i] * oMult; o0=(int) o; od= o - (float)o0;
			o0 *= nb; o0 = (oMax > o0) ? o0 : 0; O0[i]=o0;
			o1=o0 + nb; o1 = (oMax > o1) ? o1 : 0; O1[i]=o1;
			m=M[i]*norm; M1[i]= od * m; M0[i]= m - M1[i]; 
		}
#endif
	else 
#ifdef SSEv2
		for( i=0; i<=n-4; i+=4 ) {
			_o=MUL(LDu(O[i]),_oMult); _o0=CVT(ADD(_o,SET(.5f)));
			_o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
			*_M0++=MUL(LDu(M[i]),_norm); *_M1++=SET(0.f); *_O1++=SET(0);
		}
#else
		for( i=0; i<=n-4; i++ ) {
			o=O[i]*oMult; o0=(int)(o+0.5f); o0 *= nb;
			o0 = (oMax > o0) ? o0 : 0; O0[i]=o0; M0[i]=M[i]*norm;
			M1[i]=0.0f; O1[i]=0;
		}  
#endif
		// compute trailing locations without sse
		if( interpolate ) for( i; i<n; i++ ) {
			o=O[i]*oMult; o0=(int) o; od=o-o0;
			o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
			o1=o0+nb; if(o1==oMax) o1=0; O1[i]=o1;
			m=M[i]*norm; M1[i]=od*m; M0[i]=m-M1[i];
		} else for( i; i<n; i++ ) {
			o=O[i]*oMult; o0=(int) (o+.5f);
			o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
			M0[i]=M[i]*norm; M1[i]=0; O1[i]=0;
		}
}

// compute nOrients gradient histograms per bin x bin block of pixels
void gradientHist( float *M, float *O, float *H, int h, int w,
  int bin, int nOrients, int softBin, bool full )
{
  const int hb=h/bin, wb=w/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
  const float s=(float)bin, sInv=1/s, sInv2=1/s/s;
  float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float  xb, init;

#ifdef SSEv2
  O0=(int*)_aligned_malloc(h*sizeof(int),16); 
  M0=(float*)_aligned_malloc(h*sizeof(float),16);
  O1=(int*)_aligned_malloc(h*sizeof(int),16); 
  M1=(float*)_aligned_malloc(h*sizeof(float),16);
#else
  O0 = (int*)fftwf_malloc(sizeof(int)*h);
  M0 = (float*)fftwf_malloc(sizeof(float)*h);
  O1 = (int*)fftwf_malloc(sizeof(int)*h);
  M1 = (float*)fftwf_malloc(sizeof(float)*h);
#endif

  // main loop
  for( x=0; x<w0; x++ ) {
    // compute target orientation bins for entire column - very fast
    gradQuantize(O+x*h,M+x*h,O0,O1,M0,M1,nb,h0,sInv2,nOrients,full,softBin>=0);

    if( softBin<0 && softBin%2==0 ) {
      // no interpolation w.r.t. either orienation or spatial bin
      H1=H+(x/bin)*hb;
      #define GH H1[O0[y]]+=M0[y]; y++;
      if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
      else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
      else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
      else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
      else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
      #undef GH

    } else if( softBin%2==0 || bin==1 ) { 
      // interpolate w.r.t. orientation only, not spatial bin
      H1=H+(x/bin)*hb;
      #define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
      if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
      else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
      else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
      else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
      else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
      #undef GH
	}
	else { // interpolate using trilinear interpolation
		float ms[4], xyd, yb, xd, yd;
#ifdef SSEv2
		__m128 _m, _m0, _m1;
#endif
	  bool hasLf, hasRt; int xb0, yb0;
	  if (x == 0) { init = (0 + .5f)*sInv - 0.5f; xb = init; }
	  hasLf = xb >= 0; xb0 = hasLf ? (int)xb : -1; hasRt = xb0 < wb - 1;
	  xd = xb - xb0; xb += sInv; yb = init; y = 0;
      // macros for code conciseness
      #define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
        ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
#ifdef SSEv2
      #define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
#endif
      // leading rows, no top bin
      for( ; y<bin/2; y++ ) {
        yb0=-1; GHinit;
        if(hasLf) { H0[O0[y]+1]+=ms[1]*M0[y]; H0[O1[y]+1]+=ms[1]*M1[y]; }
        if(hasRt) { H0[O0[y]+hb+1]+=ms[3]*M0[y]; H0[O1[y]+hb+1]+=ms[3]*M1[y]; }
      }

      // main rows, has top and bottom bins, use SSE for minor speedup
	  if( softBin<0 ) 
		  for( ; ; y++ ) {
			  yb0 = (int) yb; if(yb0>=hb-1) break; GHinit; 
#ifdef SSEv2
			  _m0=SET(M0[y]);
#endif
			  if(hasLf) { 
#ifdef SSEv2
				  _m=SET(0,0,ms[1],ms[0]); GH(H0+O0[y],_m,_m0); 	
#else
				  H1 = H0+O0[y]; *H1 += 0*M0[y]; *(H1+1) += 0*M0[y]; 
				  *(H1+2) += ms[1]*M0[y]; *(H1+3) += ms[0]*M0[y];	
#endif	
			  }

			  if(hasRt) { 
#ifdef SSEv2
				  _m=SET(0,0,ms[3],ms[2]); GH(H0+O0[y]+hb,_m,_m0); 
#else
				  H1 = H0+O0[y]+hb; *H1 += 0*M0[y]; *(H1+1) += 0*M0[y];
				  *(H1+2) += ms[3]*M0[y]; 
				  *(H1+3) += ms[2]*M0[y];		
#endif					
			  }

		  } else for( ; ; y++ ) {
			  yb0 = (int) yb; if(yb0>=hb-1) break; GHinit;
#ifdef SSEv2
			  _m0=SET(M0[y]); 
			  _m1=SET(M1[y]);		  
#endif
			  if(hasLf) { 
#ifdef SSEv2
				  _m=SET(0,0,ms[1],ms[0]);
				  GH(H0+O0[y],_m,_m0); 	  
				  GH(H0+O1[y],_m,_m1); 			  
#else
				  H1 = H0+O0[y]; *H1 += 0*M0[y]; *(H1+1) += 0*M0[y];
				  *(H1+2) += ms[1]*M0[y]; *(H1+3) += ms[0]*M0[y];	
				  H1 = H0+O1[y]; *H1 += 0*M1[y]; *(H1+1) += 0*M1[y];
				  *(H1+2) += ms[1]*M1[y]; *(H1+3) += ms[0]*M1[y];		 	
#endif	
			  }

			  if(hasRt) { 
#ifdef SSEv2
				  _m=SET(0,0,ms[3],ms[2]);
				  GH(H0+O0[y]+hb,_m,_m0); 
				  GH(H0+O1[y]+hb,_m,_m1); 		  
#else
				  H1 = H0+O0[y]+hb; *H1 += 0*M0[y]; *(H1+1) += 0*M0[y];
				  *(H1+2) += ms[3]*M0[y]; *(H1+3) += ms[2]*M0[y];	
				  H1 = H0+O1[y]+hb; *H1 += 0*M1[y]; *(H1+1) += 0*M1[y];
				  *(H1+2) += ms[3]*M1[y]; *(H1+3) += ms[2]*M1[y];			 	
#endif	
			  }
		  }

      // final rows, no bottom bin
      for( ; y<h0; y++ ) {
        yb0 = (int) yb; GHinit;
        if(hasLf) { H0[O0[y]]+=ms[0]*M0[y]; H0[O1[y]]+=ms[0]*M1[y]; }
        if(hasRt) { H0[O0[y]+hb]+=ms[2]*M0[y]; H0[O1[y]+hb]+=ms[2]*M1[y]; }
      }
      #undef GHinit
      #undef GH
    }
  }

#ifdef SSEv2
  _aligned_free(O0);
  _aligned_free(M0);
  _aligned_free(O1);
  _aligned_free(M1);
#else
  fftwf_free(O0);
  fftwf_free(M0);
  fftwf_free(O1);
  fftwf_free(M1);
#endif

  // normalize boundary bins which only get 7/8 of weight of interior bins
  if( softBin%2!=0 ) for( int o=0; o<nOrients; o++ ) {
    x=0; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
    y=0; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
    x=wb-1; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
    y=hb-1; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
  }
}

/******************************************************************************/

// HOG helper: compute 2x2 block normalization values (padded by 1 pixel)
float* hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin) {
	float *N, *N1, *n; int o, x, y, dx, dy, hb1 = hb + 1, wb1 = wb + 1;
	float eps = 1e-4f / 4 / bin / bin / bin / bin; // precise backward equality
	N = (float*)fftwf_malloc(sizeof(float)*hb1*wb1);
	memset(N, 0, hb1*wb1*sizeof(float));
	N1 = N + hb1 + 1;
	for (o = 0; o<nOrients; o++) for (x = 0; x<wb; x++) for (y = 0; y<hb; y++)
		N1[x*hb1 + y] += H[o*wb*hb + x*hb + y] * H[o*wb*hb + x*hb + y];
	for (x = 0; x<wb - 1; x++) for (y = 0; y<hb - 1; y++) {
		n = N1 + x*hb1 + y; *n = 1 / float(sqrt(n[0] + n[1] + n[hb1] + n[hb1 + 1] + eps));
	}
	x = 0;     dx = 1; dy = 1; y = 0;                  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	x = 0;     dx = 1; dy = 0; for (y = 0; y<hb1; y++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	x = 0;     dx = 1; dy = -1; y = hb1 - 1;              N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	x = wb1 - 1; dx = -1; dy = 1; y = 0;                  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	x = wb1 - 1; dx = -1; dy = 0; for (y = 0; y<hb1; y++) N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	x = wb1 - 1; dx = -1; dy = -1; y = hb1 - 1;              N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	y = 0;     dx = 0; dy = 1; for (x = 0; x<wb1; x++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	y = hb1 - 1; dx = 0; dy = -1; for (x = 0; x<wb1; x++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
	return N;
}

// HOG helper: compute HOG or FHOG channels
void hogChannels(float *H, const float *R, const float *N,
	int hb, int wb, int nOrients, float clip, int type)
{
#define GETT(blk) t=R1[y]*N1[y-(blk)]; if(t>clip) t=clip; c++;
	const float r = .2357f; int o, x, y, c; float t;
	const int nb = wb*hb, nbo = nOrients*nb, hb1 = hb + 1;
	for (o = 0; o<nOrients; o++) for (x = 0; x<wb; x++) {
		const float *R1 = R + o*nb + x*hb, *N1 = N + x*hb1 + hb1 + 1;
		float *H1 = (type <= 1) ? (H + o*nb + x*hb) : (H + x*hb);
		if (type == 0) for (y = 0; y<hb; y++) {
			// store each orientation and normalization (nOrients*4 channels)
			c = -1; GETT(0); H1[c*nbo + y] = t; GETT(1); H1[c*nbo + y] = t;
			GETT(hb1); H1[c*nbo + y] = t; GETT(hb1 + 1); H1[c*nbo + y] = t;
		}
		else if (type == 1) for (y = 0; y<hb; y++) {
			// sum across all normalizations (nOrients channels)
			c = -1; GETT(0); H1[y] += t*.5f; GETT(1); H1[y] += t*.5f;
			GETT(hb1); H1[y] += t*.5f; GETT(hb1 + 1); H1[y] += t*.5f;
		}
		else if (type == 2) for (y = 0; y<hb; y++) {
			// sum across all orientations (4 channels)
			c = -1; GETT(0); H1[c*nb + y] += t*r; GETT(1); H1[c*nb + y] += t*r;
			GETT(hb1); H1[c*nb + y] += t*r; GETT(hb1 + 1); H1[c*nb + y] += t*r;
		}
	}
#undef GETT
}

// compute HOG features
void hog(float *M, float *O, float *H, int h, int w, int binSize,
	int nOrients, int softBin, bool full, float clip)
{
	float *N, *R; const int hb = h / binSize, wb = w / binSize, nb = hb*wb;
	//compute unnormalized gradient histograms
	R = (float*)fftwf_malloc(sizeof(float)*wb*hb*nOrients);
	memset(R, 0, wb*hb*nOrients*sizeof(float));
	gradientHist(M, O, R, h, w, binSize, nOrients, softBin, full);
	// compute block normalization values
	N = hogNormMatrix(R, nOrients, hb, wb, binSize);
	// perform four normalizations per spatial block
	hogChannels(H, R, N, hb, wb, nOrients, clip, 0);
	fftwf_free(N); fftwf_free(R);
}

// compute FHOG features
void fhog(float *M, float *O, float *H, int h, int w, int binSize,
	int nOrients, int softBin, float clip)
{
	const int hb = h / binSize, wb = w / binSize, nb = hb*wb, nbo = nb*nOrients;
	float *N, *R1, *R2; int o, x;
	// compute unnormalized constrast sensitive histograms	
	R1 = (float*)fftwf_malloc(sizeof(float)*wb*hb*nOrients * 3);//3 чтобы не падало из-за обращения в ячейку для которой не выделена память , а так должно быть 2
	memset(R1, 0, wb*hb*nOrients * 3 * sizeof(float));
	gradientHist(M, O, R1, h, w, binSize, nOrients * 2, softBin, true);
	// compute unnormalized contrast insensitive histograms	
	R2 = (float*)fftwf_malloc(sizeof(float)*wb*hb*nOrients);
	memset(R2, 0, wb*hb*nOrients * sizeof(float));
	for (o = 0; o<nOrients; o++) 
		for (x = 0; x<nb; x++)
		R2[o*nb + x] = R1[o*nb + x] + R1[(o + nOrients)*nb + x];
	// compute block normalization values
	N = hogNormMatrix(R2, nOrients, hb, wb, binSize);
	// normalized histograms and texture channels
	hogChannels(H + nbo * 0, R1, N, hb, wb, nOrients * 2, clip, 1);
	hogChannels(H + nbo * 2, R2, N, hb, wb, nOrients * 1, clip, 1);
	hogChannels(H + nbo * 3, R1, N, hb, wb, nOrients * 2, clip, 2);
	fftwf_free(R1); fftwf_free(R2); fftwf_free(N);
}
 