#####################################################################
#  rpudist.R
#    R implementation for distance and cluster functions
#
#  Based on stats/R/hclust.R and gputools/R/gpuHclust.R
#
#  Author: Chi Yau
#  Palo Alto, CA, USA
#  09/01/2010
#
#  This source code is licensed under The GNU General Public License (GPLv3)


#####################################################################
# rpuDevice
#
rpuDevice <- function() {

    deviceId <- .C("getDevice", 
                    deviceId = integer(1), 
                    PACKAGE = "rpud")$deviceId
    return (deviceId)
}


#####################################################################
# rpuDist
#
rpuDist <- function(points, method="euclidean", diag=FALSE, upper=FALSE, p=2) {

    methodNames <- c(
            "euclidean", 
            "maximum", 
            "manhattan", 
            "canberra", 
            "binary", 
            "minkowski")
    methodIndex <- pmatch(method, methodNames)     
    if(is.na(methodIndex)) {
        stop("invalid distance metric")
    }

    points <- as.matrix(points)
    num <- nrow(points)
    dim <- ncol(points)
     
    d <- .C("findDistance",
            method, 
            as.single(t(points)),
            as.integer(num),
            as.integer(dim),
            as.single(p),
            d = single(num*(num-1)/2),
            PACKAGE='rpud')$d
    attr(d, "Size")     <- num
    attr(d, "Labels")   <- dimnames(points)[[1L]]
    attr(d, "Diag")     <- diag
    attr(d, "Upper")    <- upper
    attr(d, "method")   <- method
    attr(d, "call")     <- match.call()
    if(!is.na(pmatch(method, "minkowski"))) {
        attr(d, "p")  <- p
    }
    class(d) <- "dist"

    return(d)
}



