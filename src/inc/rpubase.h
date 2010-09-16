/********************************************************************
*  rpubase.h
*  Header file for base utils
*
*  Author: Chi Yau
*  Palo Alto, CA, USA
*  08/30/2010
*
*  This source code is licensed under The GNU General Public License (GPLv3)
*********************************************************************/

#ifndef __RPUBASE_H__
#define __RPUBASE_H__


// debug macros
#ifdef _DEBUG

#define RPU_MSG(x)              printf("%s\n", (x))
#define RPU_MSG2(fmt, x)        printf((fmt), (x))

#else

#define RPU_MSG(x)              ((void)0)
#define RPU_MSG2(fmt, x)        ((void)0)

#endif     // _DEBUG


#define RPU_NON_NA(a)           (!isnan(a))
#define RPU_BOTH_NON_NA(a, b)   (!isnan(a) && !isnan(b))

#define RPU_FINITE(a)           (!isinf(a))
#define RPU_BOTH_FINITE(a, b)   (!isinf(a) && !isinf(b))

#define rpuSafeCall(err)        __rpuSafeCall((err), __FILE__, __LINE__)
#define rpuCheckMsg(msg)        __rpuCheckMsg((msg), __FILE__, __LINE__)



/********************************************************************
* RPUException
*
*/
class RPUException {

public:
    RPUException(const char* method, const char* msg, 
            cudaError err, const char* file, const int line) {
     
        sprintf(R_problem_buf, "%s(%i): %s: %s - %s.\n",
                file, line, method, msg, cudaGetErrorString(err));
    }
     
    const char* error() {
        return R_problem_buf;
    }

private:
    char R_problem_buf[R_PROBLEM_BUFSIZE];
};


inline void __rpuSafeCall(cudaError err, const char* file, const int line) {

    if( cudaSuccess != err) {
        throw new RPUException("rpuSafeCall", "CUDA runtime API error", err, file, line);
    }
}


inline void __rpuCheckMsg(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        throw new RPUException("rpuCheckMsg", errorMessage, err, file, line);
    }
#ifdef _DEBUG
    err = cudaThreadSynchronize();
    if (cudaSuccess != err) {
        throw new RPUException("rpuCheckMsg cudaThreadSynchronize", errorMessage, err, file, line);
    }
#endif
}


#endif     // __RPUBASE_H__



