\name{rpuDist}
\alias{rpuDist}
\title{Compute the distance matrix with GPU}

\description{
     This function computes the distance between each vector of the
     'points' argument using the metric specified by 'method'.
}

\usage{
     rpuDist(points, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)
}

\arguments{
     \item{points}{a matrix of floating point numbers in which each row is a
          vector in $R^n$ space where $n$ is ncol(points).}
     \item{method}{a string representing the name of the metric to use to
          calculate the distance between the vectors of 'points'.  Currently
          supported values are: \code{"binary"}, \code{"canberra"}, 
          \code{"euclidean"}, \code{"manhattan"}, \code{"maximum"}, and 
          \code{"minkowski"}.}
     \item{diag}{logical value indicating whether the diagonal of the
          distance matrix should be printed by \code{print.dist}.}
     \item{upper}{logical value indicating whether the upper triangle of the
          distance matrix should be printed by \code{print.dist}.}
     \item{p}{The power of the Minkowski distance.}
}

\value{
     a class of type "dist" containing floating point numbers representing the 
     distances between vectors from the 'points' argument.
}

\seealso{
     dist, gpuDist
}

\examples{
numVectors <- 5
dimension <- 10
Vectors <- matrix(runif(numVectors*dimension), numVectors, dimension)
rpuDist(Vectors, "euclidean")
rpuDist(Vectors, "maximum")
rpuDist(Vectors, "manhattan")
rpuDist(Vectors, "canberra")
rpuDist(Vectors, "binary")
rpuDist(Vectors, "minkowski")
}

\keyword{math, array & algebra}


