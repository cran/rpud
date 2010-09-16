/********************************************************************
*  rputypes.h
*  Header file for common data types
*
*  Author: Chi Yau
*  Palo Alto, CA, USA
*  08/30/2010
*
*  This source code is licensed under The GNU General Public License (GPLv3)
*********************************************************************/

#ifndef __RPUTYPES_H__
#define __RPUTYPES_H__

#ifdef __cplusplus
extern "C"
{
#endif 


typedef int         Integer;
typedef Integer*    PInteger;

typedef char*       String;
typedef String*     PString;

typedef float       Numeric;
typedef Numeric*    PNumeric;


#define RPU_NUMERIC_MAX     FLT_MAX
#define RPU_NUMERIC_MIN     FLT_MIN



#ifdef __cplusplus
}
#endif 

#endif     // __RPUTYPES_H__



