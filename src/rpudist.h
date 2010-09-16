/********************************************************************
*  rpudist.h
*  Header file for rpudist
*
*  Author: Chi Yau
*  Palo Alto, CA, USA
*  08/30/2010
*
*  This source code is licensed under The GNU General Public License (GPLv3)
*********************************************************************/

#ifndef __RPUDIST_H__
#define __RPUDIST_H__


#include "inc/rputypes.h"
#include "inc/rpubase.h"


// calculate the offset in the distance array
#define RPU_DISOFFST(n, x, y)          ((y)+(x)*(n)-((x)+1)*((x)+2)/2)



/********************************************************************
* RPUDist
*
*/
class RPUDist {

public:
    RPUDist(const String type, const PNumeric input, 
                const Integer num, const Integer dim, const Numeric power);
     
    void find(PNumeric output);

protected:
    const PNumeric 
        m_input;
    const Integer
        m_nx, m_ny, m_num;
    const Integer
        BLK_DX, BLK_DY;          // block dimensions
    const Numeric
        m_power;
         
    void (*m_kernel)(PNumeric, Integer, Integer, Numeric);

};


#endif     // __RPUDIST_H__




