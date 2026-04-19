\name{plot.partialpro}
\alias{plot.partialpro}
\title{Plot method for partialpro objects}

\description{
 Plot partial effects from a previous \code{partialpro} analysis.
}

\usage{\method{plot}{partialpro}(x, xvar.names, nvar,
  parametric = FALSE, se = TRUE,
  causal = FALSE, subset = NULL, plot.it = TRUE, ...)
}

\arguments{

  \item{x}{A \code{partialpro} object returned from a previous call to
    \code{partialpro}.}

  \item{xvar.names}{Names (or integer indices) of the x-variables to
    plot. Defaults to all variables.}

  \item{nvar}{Number of variables to plot. Defaults to all variables.}

  \item{parametric}{Logical. Set to \code{TRUE} only if the partial
    effect is believed to follow a polynomial form.}

  \item{se}{Display standard errors?}

  \item{causal}{Display causal estimator?}

  \item{subset}{Optional conditioning factor. Not applicable if
    \code{parametric = TRUE}. May also be a logical or integer vector to
    subset the analysis.}

  \item{plot.it}{If \code{FALSE}, no plot is produced; instead, the
    internal plotting objects are returned.}

  \item{...}{Additional arguments passed to \code{plot}.}

}

\details{

Generates smoothed partial-effect plots for continuous variables. The
solid black line represents the estimated partial effect; dashed red
lines show an approximate plus-minus standard error band. These
standard errors are intended as heuristic guides and should be
interpreted cautiously.

Partial effects are estimated nonparametrically using locally fitted
polynomial models. This is the default behavior and is recommended when
effects are expected to be nonlinear. Use \code{parametric = TRUE} if
the underlying effect is believed to follow a global polynomial form.

For binary variables, partial effects are shown as boxplots, with
whiskers reflecting variability analogous to standard error.

The causal estimator, when requested, displays the baseline-subtracted
local effect.

Conditioning is supported via the \code{subset} option. When supplied as
a factor (with length equal to the original data), the plot is
stratified by its levels. Alternatively, \code{subset} can be a logical
or integer vector indicating the cases to include in the analysis.

}

\value{

If \code{plot.it = TRUE}, the method is called for its side effect of
producing plots and returns \code{NULL}.

If \code{plot.it = FALSE}, the method returns a named list of internal
plot objects, one per requested variable, containing the partial-effect
curves and associated summaries used for plotting.

}

\author{

  Min Lu and Hemant Ishwaran

}

\references{

  Ishwaran H. (2025). Multivariate Statistics: Classical Foundations
  and Modern Machine Learning, CRC (Chapman and Hall), in press.

}

\seealso{
  \command{\link{partialpro}}
}

\examples{
\donttest{
##------------------------------------------------------------------
##
## Boston housing
##
##------------------------------------------------------------------

library(mlbench)
data(BostonHousing)
o.boston <- varpro(medv~., BostonHousing)
oo.boston <- partialpro(o.boston, nvar = 4, learner = rf.learner(o.boston))

oldpar <- par(mfrow = c(2, 4))

## parametric local estimation
plot(oo.boston, parametric = TRUE, ylab = "parametric est.")

## non-parametric local estimation (default)
plot(oo.boston, parametric = FALSE, ylab = "non-parametric est.")

par(oldpar)

##------------------------------------------------------------------
##
## Boston housing with subsetting
##
##------------------------------------------------------------------

library(mlbench)
data(BostonHousing)
o.boston <- varpro(medv~., BostonHousing)
oo.boston <- partialpro(o.boston, nvar = 3, learner = rf.learner(o.boston))

## subset analysis
price <- BostonHousing$medv
pricef <- factor(price > median(price), labels = c("low priced", "high priced"))
oldpar <- par(mfrow = c(1, 1))
plot(oo.boston, subset = pricef, nvar = 1)
par(oldpar)

##------------------------------------------------------------------
##
## veteran data with subsetting using celltype as a factor
##
##------------------------------------------------------------------

data(veteran, package = "randomForestSRC")
dta <- veteran
dta$celltype <- factor(dta$celltype)
o.vet <- varpro(Surv(time, status) ~ ., dta)
oo.vet <- partialpro(o.vet, nvar = 6, nsmp = Inf, learner = rf.learner(o.vet))

## partial effects, with subsetting
oldpar <- par(mfrow = c(2, 3))
plot(oo.vet, subset = dta$celltype)
par(oldpar)

## causal effects, with subsetting
oldpar <- par(mfrow = c(2, 3))
plot(oo.vet, subset = dta$celltype, causal = TRUE)
par(oldpar)

## retrieve plotting objects without drawing
obj <- plot(oo.vet, subset = dta$celltype, plot.it = FALSE)
str(obj, max.level = 1)

}
}
\keyword{plot}
