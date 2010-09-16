/********************************************************************
*  rpudist.cu
*  Header file for rpudist
*
*  Author: Chi Yau
*  Palo Alto, CA, USA
*  08/30/2010
*
*  This source code is licensed under The GNU General Public License (GPLv3)
*********************************************************************/


#include <R.h>
#include "rpudist.h"


static texture<Numeric, 2, cudaReadModeElementType> textRef;     // 2D texture


/********************************************************************
* EuclideanMetric
*
*/
__global__ void euclideanMetricKernel(
    PNumeric d_output, Integer dim, Integer num, Numeric power) {

    const int 
        xid = blockIdx.x * blockDim.x + threadIdx.x,
        yid = blockIdx.y * blockDim.y + threadIdx.y;
        
    // compute the distance between the x-th and y-th vectors
    if (xid < yid && yid < num) {

        Integer count = 0;
        Numeric sum = 0;
        for (int i = 0; i < dim; i++) {
            const Numeric dev = 
                tex2D(textRef, i, xid) - 
                tex2D(textRef, i, yid);
            if (RPU_NON_NA(dev)) {
                sum += dev * dev;
                count++;
            }
        }
         
        if (count != dim) {
            sum /= ((Numeric)count/dim);
        }

        d_output[RPU_DISOFFST(num, xid, yid)] = sqrt(sum);
     }
}


/********************************************************************
* MaximumMetric
*
*/
__global__ void maximumMetricKernel(
    PNumeric d_output, Integer dim, Integer num, Numeric power) {

    const int 
        xid = blockIdx.x * blockDim.x + threadIdx.x,
        yid = blockIdx.y * blockDim.y + threadIdx.y;
        
    // compute the distance between the x-th and y-th vectors
    if (xid < yid && yid < num) {

        Integer count = 0;
        Numeric dist = -RPU_NUMERIC_MAX;
        for (int i = 0; i < dim; i++) {
            const Numeric
                x = tex2D(textRef, i, xid),
                y = tex2D(textRef, i, yid);
            if (RPU_BOTH_NON_NA(x, y)) {
                const Numeric dev = abs(x - y);
                if (RPU_NON_NA(dev)) {
                    if (dist < dev) dist = dev;
                    count++;
                }
            }
        }

        if (count != dim) {
            dist /= ((Numeric)count/dim);
        }

        d_output[RPU_DISOFFST(num, xid, yid)] = dist;
     }
}


/********************************************************************
* ManhattanMetric
*
*/
__global__ void manhattanMetricKernel(
    PNumeric d_output, Integer dim, Integer num, Numeric power) {

    const int 
        xid = blockIdx.x * blockDim.x + threadIdx.x,
        yid = blockIdx.y * blockDim.y + threadIdx.y;
        
    // compute the distance between the x-th and y-th vectors
    if (xid < yid && yid < num) {

        Integer count = 0;
        Numeric dist = 0;
        for (int i = 0; i < dim; i++) {
            const Numeric
                x = tex2D(textRef, i, xid),
                y = tex2D(textRef, i, yid);
            if (RPU_BOTH_NON_NA(x, y)) {
                const Numeric dev = abs(x - y);
                if (RPU_NON_NA(dev)) {
                    dist += dev;
                    count++;
                 }
            }
        }

        if (count != dim) {
            dist /= ((Numeric)count/dim);
        }

        d_output[RPU_DISOFFST(num, xid, yid)] = dist;
    }
}


/********************************************************************
* CanberraMetric
*
*/
__global__ void canberraMetricKernel(
    PNumeric d_output, Integer dim, Integer num, Numeric power) {

    const int 
        xid = blockIdx.x * blockDim.x + threadIdx.x,
        yid = blockIdx.y * blockDim.y + threadIdx.y;
        
    // compute the distance between the x-th and y-th vectors
    if (xid < yid && yid < num) {

        Integer count = 0;
        Numeric dist = 0;
        for (int i = 0; i < dim; i++) {
            const Numeric
                x = tex2D(textRef, i, xid),
                y = tex2D(textRef, i, yid);
            if (RPU_BOTH_NON_NA(x, y)) {
                const Numeric 
                    sum = abs(x + y),   // [Bug 14375]
                    diff = abs(x - y);
                if (sum > RPU_NUMERIC_MIN || diff > RPU_NUMERIC_MIN) {
                    Numeric dev = diff/sum;
                    if (RPU_NON_NA(dev) || 
                        (!RPU_FINITE(diff) && diff == sum && 
                            /* use Inf = lim x -> oo */ (dev = 1.))) {
                            dist += dev;
                            count++;
                    }
                }
            }
        }

        if (count != dim) {
            dist /= ((Numeric)count/dim);
        }

        d_output[RPU_DISOFFST(num, xid, yid)] = dist;
    }
}


/********************************************************************
* BinaryMetric
*
*/
__global__ void binaryMetricKernel(
    PNumeric d_output, Integer dim, Integer num, Numeric power) {

    const int 
        xid = blockIdx.x * blockDim.x + threadIdx.x,
        yid = blockIdx.y * blockDim.y + threadIdx.y;
        
    // compute the distance between the x-th and y-th vectors
    if (xid < yid && yid < num) {

        Integer count = 0;
        Numeric dist = 0;
        for (int i = 0; i < dim; i++) {
            const Numeric
                x = tex2D(textRef, i, xid),
                y = tex2D(textRef, i, yid);
            if (RPU_BOTH_NON_NA(x, y) && RPU_BOTH_FINITE(x, y)) {
                if (x || y) {
                    count++;
                    if (!(x && y)) dist++;
                }
            }
        }

        if (count != dim) {
            dist /= ((Numeric)count/dim);
        }

        d_output[RPU_DISOFFST(num, xid, yid)] = dist;
    }
}


/********************************************************************
* MinkowskiMetric
*
*/
__global__ void minkowskiMetricKernel(
    PNumeric d_output, Integer dim, Integer num, Numeric power) {

    const int 
        xid = blockIdx.x * blockDim.x + threadIdx.x,
        yid = blockIdx.y * blockDim.y + threadIdx.y;
        
    // compute the distance between the x-th and y-th vectors
    if (xid < yid && yid < num) {

        Integer count = 0;
        Numeric sum = 0;
        for (int i = 0; i < dim; i++) {
            const Numeric
                x = tex2D(textRef, i, xid),
                y = tex2D(textRef, i, yid);
            if (RPU_BOTH_NON_NA(x, y)) {
                const Numeric dev = (x - y);
                if (RPU_NON_NA(dev)) {
                    sum += __powf(abs(dev), power);
                    count++;
                }
            }
        }

        if (count != dim) {
            sum /= ((Numeric)count/dim);
        }

        d_output[RPU_DISOFFST(num, xid, yid)] = __powf(sum, 1.0/power);
    }
}



/********************************************************************
* RPUDist::RPUDist
*
*/
RPUDist::RPUDist(const String type, const PNumeric input, 
                    const Integer num, const Integer dim, const Numeric power) :
            m_input(input), m_nx(dim), m_ny(num), m_num(num), 
            m_power(power), BLK_DX(16), BLK_DY(16) {

    m_kernel =
        (!strcmp("euclidean", type))    ? ::euclideanMetricKernel   :
        (!strcmp("maximum", type))      ? ::maximumMetricKernel     :
        (!strcmp("manhattan", type))    ? ::manhattanMetricKernel   :
        (!strcmp("canberra", type))     ? ::canberraMetricKernel    :
        (!strcmp("binary", type))       ? ::binaryMetricKernel      :
        (!strcmp("minkowski", type))    ? ::minkowskiMetricKernel   :
        NULL;
}



/********************************************************************
* RPUDist::find
*
*/
void RPUDist::find(PNumeric h_output) {

    RPU_MSG("RPUDist::find starts");
     
    cudaArray* cuArray = NULL;
    PNumeric d_output = NULL;
     
    try {
        ///////////////////////////////////////////////////////////////////
        // device input
        const size_t INPUT_SIZE = sizeof(Numeric)*(m_nx)*(m_ny);
        cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<Numeric>();
        rpuSafeCall(cudaMallocArray(&cuArray, &chDesc, m_nx, m_ny));

        // copy input from host to device
        rpuSafeCall(cudaMemcpyToArray(cuArray, 0, 0, m_input, INPUT_SIZE, cudaMemcpyHostToDevice));

        // bind texture to array
        rpuSafeCall(cudaBindTextureToArray(::textRef, cuArray));

        ///////////////////////////////////////////////////////////////////
        // device output
        const size_t OUTPUT_SIZE = sizeof(Numeric)*(m_num)*(m_num-1)/2;
        rpuSafeCall(cudaMalloc((void**)&d_output, OUTPUT_SIZE));
    
        ///////////////////////////////////////////////////////////////////
        // kernel call
        if (m_kernel != NULL) {
            const dim3 
                grid((m_num+BLK_DX-1)/BLK_DX, (m_num+BLK_DY-1)/BLK_DY), 
                blk(BLK_DX, BLK_DY);
            m_kernel<<<grid, blk>>>(d_output, m_nx, m_ny, m_power);
            rpuCheckMsg("Kernel execution failed");

            // block until the device has completed
            rpuSafeCall(cudaThreadSynchronize());

            // copy device output to host
            rpuSafeCall(cudaMemcpy(h_output, d_output, OUTPUT_SIZE, cudaMemcpyDeviceToHost));
     
        } else {
          warning("Unsupported distance type");
        }
    
        RPU_MSG("RPUDist::find ends");
     
    } catch (RPUException* e) {
     
        error(e->error());
        delete e;
    }

    ///////////////////////////////////////////////////////////////////
    // free device memory
    cudaUnbindTexture(::textRef);
    if (cuArray != NULL) cudaFreeArray(cuArray);
    if (d_output != NULL) cudaFree(d_output);
}






