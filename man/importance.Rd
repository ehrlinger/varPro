\name{importance}
\alias{importance}
\alias{importance.varpro}
\alias{importance.uvarpro}
\alias{importance.rhf}

\title{Calculate Importance for VarPro and UVarPro Objects}

\description{
Calculates variable importance from compatible objects such as \code{varpro}
and \code{uvarpro}.
}

\usage{
importance(x, ...)

\method{importance}{varpro}(x, local.std = TRUE, y.external = NULL,
  cutoff = 0.79, trim = 0.1, plot.it = FALSE, conf = TRUE, sort = TRUE,
  ylab = if (conf) "Importance" else "Standardized Importance",
  max.rules.tree, max.tree,
  ...)

\method{importance}{uvarpro}(x, local.std = FALSE, y.external = NULL,
  cutoff = 0.79, trim = 0.1, plot.it = FALSE, conf = TRUE, sort = TRUE,
  ylab = if (conf) "Importance" else "Standardized Importance",
  max.rules.tree, max.tree,
  ...)

\method{importance}{rhf}(x, local.std = TRUE, y.external = NULL,
  cutoff = 0.79, trim = 0.1, plot.it = FALSE, conf = TRUE, sort = TRUE,
  ylab = if (conf) "Importance" else "Standardized Importance",
  max.rules.tree, max.tree,
  ...)


}

\arguments{

  \item{x}{A \code{varpro}, \code{uvarpro} or \code{rhf} object.}

  \item{local.std}{Logical. If \code{TRUE}, uses locally standardized
  importance values. Ignored for \code{uvarpro} objects.}

  \item{y.external}{Optional user-supplied response vector. Must match the
  expected dimension and outcome family. Ignored for \code{uvarpro} objects.}

  \item{cutoff}{Threshold used to highlight significant variables in the
  importance plot. Applies only when \code{plot.it = TRUE}.}

  \item{trim}{Windsorization trim value used to robustify the mean and standard
  deviation calculations.}

  \item{plot.it}{Logical. If \code{TRUE}, generates a plot of importance
  values.}

  \item{conf}{Logical. If \code{TRUE}, displays importance values with standard
  errors as a boxplot (providing an informal confidence region). If
  \code{FALSE}, plots standardized importance values.}

  \item{sort}{Logical. If \code{TRUE}, sorts results in decreasing order of
  importance.}

  \item{ylab}{Character string specifying the y-axis label.}

  \item{max.rules.tree}{Optional. Maximum number of rules per tree. Defaults to
  the value stored in the fitted object if unspecified.}

  \item{max.tree}{Optional. Maximum number of trees used for rule extraction.
  Defaults to the value from the fitted object if unspecified.}

  \item{...}{Additional arguments passed to internal methods.}

}

\details{

This page documents the public \code{importance()} generic together with the
methods for \code{varpro} and \code{uvarpro} objects.

The supervised \code{varpro} method calculates standardized importance values
for identifying and ranking variables. Optionally, graphical output is
provided, including confidence-style boxplots.

}


\value{

  Invisibly, a table summarizing the results. Contains mean importance
  \code{mean}, the standard deviation \code{std}, and standardized importance
  \code{z}.

  For classification, conditional \code{z} tables are additionally provided,
  where the \eqn{z} standardized importance values are conditional on the class
  label.

  See \command{cv.varpro} for a data-driven cross-validation method for
  selecting the cutoff value, \code{cutoff}, in supervised
  \code{varpro} analyses.

}

\author{

  Min Lu and Hemant Ishwaran

}

\references{

  Lu, M. and Ishwaran, H., (2024). Model-independent variable selection
  via the rule-based variable priority. arXiv e-prints, pp.arXiv-2409.


}

\seealso{
  \command{\link{cv.varpro}}
  \command{\link{varpro}}
  \command{\link{uvarpro}}
}

\examples{


## ------------------------------------------------------------
## toy example - needed to pass CRAN test
## ------------------------------------------------------------

## mtcars regression
o <- varpro(mpg ~ ., mtcars, ntree = 1)
imp <- importance(o, local.std = FALSE)
print(imp)

\donttest{

## ------------------------------------------------------------
## iris example
## ------------------------------------------------------------

## apply varpro to the iris data
o <- varpro(Species ~ ., iris, max.tree = 5)

## print/plot the results
imp <- importance(o, plot.it = TRUE)
print(imp)

## ------------------------------------------------------------
## boston housing: regression
## ------------------------------------------------------------

data(BostonHousing, package = "mlbench")

## call varpro
o <- varpro(medv~., BostonHousing)

## extract importance values
imp <- importance(o)
print(imp)

## plot the results
imp <- importance(o, plot.it = TRUE)
print(imp)



## ------------------------------------------------------------
## illustrates y-external: regression example
## ------------------------------------------------------------

## friedman1 - standard application of varpro
d <- data.frame(mlbench::mlbench.friedman1(250),noise=matrix(runif(250*10,-1,1),250))
o <- varpro(y~.,d)
print(importance(o))

## importance using external rf predictor
print(importance(o,y.external=randomForestSRC::rfsrc(y~.,d)$predicted.oob))

## importance using external lm predictor
print(importance(o,y.external=lm(y~.,d)$fitted))

## importance using external randomized predictor
print(importance(o,y.external=sample(o$y)))

## ------------------------------------------------------------
## illustrates y-external: classification example
## ------------------------------------------------------------

## iris - standard application of varpro
o <- varpro(Species~.,iris)
print(importance(o))

## importance using  external rf predictor
print(importance(o,y.external=randomForestSRC::rfsrc(Species~.,iris)$class.oob))

## importance using  external randomized predictor
print(importance(o,y.external=sample(o$y)))

## ------------------------------------------------------------
## illustrates y-external: survival
## ------------------------------------------------------------
data(pbc, package = "randomForestSRC")
o <- varpro(Surv(days, status)~., pbc)
print(importance(o))

## importance using  external rsf predictor
print(importance(o,y.external=randomForestSRC::rfsrc(Surv(days, status)~., pbc)$predicted.oob))

## importance using  external randomized predictor
print(importance(o,y.external=sample(o$y)))


}
}
\keyword{varpro}
