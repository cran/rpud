/********************************************************************
*  rpud.cu
*  Implementation file for rpud
*
*  Author: Chi Yau
*  Palo Alto, CA, USA
*  08/30/2010
*
*  This source code is licensed under The GNU General Public License (GPLv3)
*********************************************************************/

#include <R.h>

#include "rpud.h"
#include "rpudist.h"



/********************************************************************
* getDevice
*
*/
void getDevice(PInteger device) {

    try {
        rpuSafeCall(cudaGetDevice(device));

    } catch (RPUException* e) {
        error(e->error());
        delete e;
    }
}


/********************************************************************
* findDistance
*
*/
void findDistance(
        const PString pType, 
        const PNumeric points, 
        const PInteger pNum, 
        const PInteger pDim,
        const PNumeric pMinkowski,
        PNumeric pDistances) {

    String type = *pType;
    RPU_MSG2("Finding %s distances ...\n", type);
     
    RPUDist dist(type, points, *pNum, *pDim, *pMinkowski);
    dist.find(pDistances);
}



