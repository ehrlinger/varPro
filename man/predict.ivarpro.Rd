\name{predict.ivarpro}
\alias{predict.ivarpro}
\title{Predict iVarPro Gradients on New Data}

\description{

  Calculates case-specific iVarPro gradients on a new feature matrix by
  reusing the rule-level gradients stored in a previously computed
  iVarPro object and recomputing rule membership for the new cases.

}

\usage{
predict.ivarpro(object,
              newdata = NULL,
              model = NULL,
              noise.na = NULL,
              path.store.membership = FALSE,
              save.data = TRUE,
              ...)
}

\arguments{

  \item{object}{An object returned by \code{\link{ivarpro}} (a numeric
    \code{data.frame} for univariate outcomes, or a list of such
    \code{data.frame}s for multivariate outcomes). The object must contain the
    \code{"ivarpro.path"} attribute with stored rule metadata and rule-level
    gradients.}

  \item{newdata}{Optional data.frame of predictors for which to compute
    case-specific gradients. If supplied, gradients are computed for
    \code{newdata} using full membership (all eligible trees). If
    \code{NULL} (default), the function attempts a restore prediction
    for the training data, using OOB membership so that the returned
    gradients match the original \code{ivarpro()} output when possible.}

  \item{model}{Optional model override. If \code{NULL} (default), the
    model is taken from \code{attr(object, "model")}. The stored model
    can be either a \code{varpro} object or a \code{randomForestSRC}
    \code{rfsrc} grow object.}

  \item{noise.na}{Logical controlling how cells with no usable
    contributing rules are handled. If \code{NULL} (default), inherits
    the setting stored in \code{attr(object,
    "ivarpro.path")$noise.na}. If \code{TRUE}, such cells are set to
    \code{NA}; if \code{FALSE}, they are set to zero.}

  \item{path.store.membership}{Logical. If \code{TRUE}, store the rule
    membership indices (case IDs) for the prediction in \code{attr(out,
    "ivarpro.path")$oobMembership}. This enables ladder-based bands in
    \code{\link{partial.ivarpro}} and summaries via
    \code{ivarpro_band()} for the predicted gradients, at the cost of
    additional memory. Default is \code{FALSE}.}

  \item{save.data}{Logical. If \code{TRUE} (default) and \code{newdata}
    is supplied, save the predictor data used for prediction as
    \code{attr(out, "data")}, enabling downstream plotting functions
    (e.g., \code{\link{partial.ivarpro}}) to retrieve \code{x}
    automatically.}

  \item{...}{Additional arguments passed to
    \code{randomForestSRC::predict.rfsrc()}.}

}

\details{

A previously computed iVarPro object contains (i) a set of retained
VarPro rules identified by tree and node/branch IDs, (ii) a
release-variable index for each rule, and (iii) an estimated rule-level
gradient. For new cases, the function obtains terminal node membership
for each tree using \code{randomForestSRC::predict.rfsrc(membership =
TRUE)} and determines which stored rules apply to each case by matching
the case's terminal node to the stored rule node/branch ID for that
tree. Case-specific gradients are then computed by aggregating the
stored rule-level gradients over all rules that apply to the case and
release the corresponding variable.

When \code{newdata} is not supplied, \code{predict.ivarpro()} attempts
to return OOB case-specific gradients for the original training
data. When the iVarPro object was created with membership information
stored in its path (\code{path.store.membership = TRUE}, the default for
\code{ivarpro()}), the restore prediction will reproduce the original
iVarPro matrix.

If \code{save.data = TRUE} and \code{newdata} is supplied,
\code{attr(out,"data")} contains the predictor matrix used for
prediction, so downstream wrappers such as \code{\link{partial.ivarpro}}
can be used directly. If you want ladder bands on predicted gradients,
set \code{path.store.membership = TRUE} when predicting.

}

\value{

Returns an object with the same shape as \code{object}:

\itemize{
  
  \item For univariate outcomes, a numeric \code{data.frame} of
  dimension \eqn{n_{test} \times p} containing predicted
  case-specific gradients.

  \item For multivariate outcomes, a named list of such
  \code{data.frame}s, one per outcome coordinate.

}

The returned object includes an \code{"ivarpro.path"} attribute
containing rule metadata and (optionally) prediction-time membership
indices.

}

\seealso{
  \code{\link{ivarpro}},
  \code{\link{partial.ivarpro}},
  \code{\link{varpro}}
}

\examples{
\donttest{

## ------------------------------------------------------------
##
## Restore mode example (training OOB gradients are reproduced)
##
## ------------------------------------------------------------

data(peakVO2, package = "randomForestSRC")

## Fit VarPro and iVarPro
vp  <- varpro(Surv(ttodead, died) ~ ., peakVO2, ntree = 50)
imp <- ivarpro(vp)

## Restore prediction: should match the original iVarPro matrix
p.restore <- predict.ivarpro(imp)
all.equal(p.restore, imp, check.attributes = FALSE)

## Downstream wrapper: partial plot using the restored gradients
partial.ivarpro(p.restore, var = "peak.vo2",
    col.var = "interval", size.var = "y", x = attr(imp, "data"))


## ------------------------------------------------------------
##
## Synthetic example (Friedman #1) with prediction on new data
##
## ------------------------------------------------------------

if (requireNamespace("mlbench", quietly = TRUE)) {

  set.seed(123)

  ## training data
  tr <- mlbench::mlbench.friedman1(500, sd = 1)
  train <- data.frame(tr$x, y = tr$y)
  colnames(train)[1:ncol(tr$x)] <- paste0("x", 1:ncol(tr$x))

  ## ivarpro fit on training data
  vp <- varpro(y ~ ., train, ntree = 100)
  imp <- ivarpro(vp)

  ## test data
  te <- mlbench::mlbench.friedman1(1e4, sd = 1)
  test <- data.frame(te$x)
  colnames(test) <- paste0("x", 1:ncol(te$x))

  ## predicted gradients on test data
  p.test <- predict.ivarpro(imp, newdata = test)

  ## partial plot directly on predicted object
  partial.ivarpro(p.test, var = "x1", col.var = "x2")


  ## push x1 outside the original Friedman support [0, 1]
  ## a heuristic way to test out-of-distribution (OOD)
  test.ood <- test
  test.ood$x1 <- runif(nrow(test), min = -0.5, max = 1.5)

  ## predicted gradients on test data
  p.test.ood <- predict.ivarpro(imp, newdata = test.ood)

  ## partial plot showing support for x1
  partial.ivarpro(p.test.ood, var = "x1", col.var = "x2",
             x.dist = c("density", "rug"))
  ## reference lines for the original training support of x1
  abline(v = c(0, 1), lty = 2)

}

}}

\keyword{individual importance}
