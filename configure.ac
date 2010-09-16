AC_INIT([rpud], 0.0.2)
AC_CONFIG_AUX_DIR([tools])

AC_ARG_WITH([cuda-home],
     AC_HELP_STRING([--with-cuda-home=cuda_home],
          [specify the root of your cuda install, e.g. /usr/local/cuda]),
     [CUDA_HOME="$withval"])

AC_ARG_WITH([r-include],
     AC_HELP_STRING([--with-r-include=r_include],
          [specify the full path to your R header files, e.g. /usr/share/R/include]),
     [R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
     AC_HELP_STRING([--with-r-lib=r-lib],
          [specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
     [R_LIB="$withval"])

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
     AC_MSG_RESULT("CUDA_HOME not set; using default /usr/local/cuda")
     CUDA_HOME="/usr/local/cuda"
else
     AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
     AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
     AC_MSG_RESULT("nvcc found")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
     AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
     CUDA_LIB_DIR="/lib"
else
     AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
     CUDA_LIB_DIR="/lib64"
fi

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
     AC_MSG_ERROR("could not determine R_HOME")
else
     CPICFLAGS=`"${R_HOME}/bin/R" CMD config CPICFLAGS`
     if test -z "${R_INCLUDE}"; then
          R_INCLUDE=`"${R_HOME}/bin/R" CMD config --cppflags`
     else
          R_INCLUDE="-I${R_INCLUDE}"
     fi
     if test -z "${R_LIB}"; then
          R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
     else
          R_LIB="-L${R_LIB} -lR"
     fi
     AC_MSG_RESULT("using ${R_INCLUDE} for R header files")
     AC_MSG_RESULT("using ${R_LIB} for R shared libraries")
fi

LIBS="-L${CUDA_HOME}${CUDA_LIB_DIR} -lcublas -lcuda"
RPATHFLAG="-Wl,-rpath,${CUDA_HOME}${CUDA_LIB_DIR}"

AC_SUBST(LIBS)
AC_SUBST(CUDA_HOME)

AC_SUBST(R_INCLUDE)
AC_SUBST(R_LIB)

AC_SUBST(CPICFLAGS)
AC_SUBST(RPATHFLAG)

AC_CONFIG_FILES([src/Makefile])
AC_OUTPUT
