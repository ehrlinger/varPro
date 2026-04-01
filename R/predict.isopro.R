#' Predict from isopro
#'
#' Scores new observations using isolation-style case depth.
#'
#' @param object An isopro object.
#' @param newdata Optional data to score.
#' @param quantiles Logical; return quantiles instead of raw depth.
#' @param ... Additional prediction arguments.
#'
#' @return Numeric vector of scores.
#' @method predict isopro
#' @export
predict.isopro <- function(object, newdata, quantiles = TRUE, ...) {
  ## must be an isopro object
  if (!inherits(object, "isopro")) {
    stop("object must be an 'isopro' varpro object")
  }
  ## if test data missing revert to original data
  if (missing(newdata)) {
    newdata <- object$isoforest$xvar
  }
  ## convert data to a data fram
  if (!is.data.frame(newdata)) {
    newdata <- data.frame(newdata)
  }
  ## test case depth values
  test.case.depth <- colMeans(predict.rfsrc(object$isoforest,
            newdata, case.depth = TRUE)$case.depth, na.rm = TRUE)
  ## return the howbad quantile
  if (quantiles) {
    object$cdf(test.case.depth)
  }
  else {
    test.case.depth
  }
}
