
#ifndef _DATAWRAPPER_H_
#define _DATAWRAPPER_H_

#include <assert.h>
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;

namespace predrecon
{

class DataWrapper{
    private: 
    double*   	     data;
    int       		 npoints;
    const static int ndim = 3; 
        
    public:
        
    void factory( double* data, int npoints ){
        this->data 	  = data;
        this->npoints = npoints;
    }
    /** 
     *  Data retrieval function
     *  @param a address over npoints
     *  @param b address over the dimensions
     */
    inline double operator()(int a, int b){
        assert( a<npoints );
        assert( b<ndim );
        return data[ a + npoints*b ];
    }
    // retrieve a single point at offset a, in a vector (preallocated structure)
    inline void operator()(int a, vector<double>& p){
        assert( a<npoints );
        assert( (int)p.size() == ndim );
        p[0] = data[ a + 0*npoints ];
        p[1] = data[ a + 1*npoints ];
        p[2] = data[ a + 2*npoints ];
    }
    int length(){
        return this->npoints;
    }
};

}

#endif