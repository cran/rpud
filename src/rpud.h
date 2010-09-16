/********************************************************************
*  rpud.h
*  Header file for rpud
*
*  Author: Chi Yau
*  Palo Alto, CA, USA
*  08/30/2010
*
*  This source code is licensed under The GNU General Public License (GPLv3)
*********************************************************************/

#ifndef __RPUD_H__
#define __RPUD_H__

#ifdef __cplusplus
extern "C"
{
#endif 

#include "inc/rputypes.h"


/********************************************************************
* getDevice
*
* @param device               Device Id of the GPU in use by the current thread
*
*********************************************************************/
void getDevice(PInteger device);



/********************************************************************
* findDistance
*
* @param pType               Type of distance metric
* @param points               Address of the input row vector matrix
* @param pNum               Number of row vectors in the matrix
* @param pDim               Dimension of each row vector in the matrix
* @param pMinkowski          Minkowski dimension
* @param pDistances          Distance matrix of the vectors
*
*********************************************************************/
void findDistance(
        const PString pType, 
        const PNumeric points, 
        const PInteger pNum, 
        const PInteger pDim,
        const PNumeric pMinkowski,
        PNumeric pDistances);


#ifdef __cplusplus
}
#endif 

#endif     // __RPUD_H__







