#' Prediction for Isopro for Identifying Anomalous Data
#' 
#' Use isolation forests to identify rare/anomalous values using test data.
#' 
#' 
#' Uses a previously constructed \code{isopro} object to assess anomalous
#' observations in the test data. By default, returns quantile values
#' representing the depth of each test observation relative to the original
#' training data. Smaller values indicate greater outlyingness.
#' 
#' To return raw depth values instead of quantiles, set \code{quantiles =
#' FALSE}.
#' 
#' @param object \code{isopro} object returned from a previous call.
#' @param newdata Optional test data. If not provided, the training data is
#' used.
#' @param quantiles Logical. If \code{TRUE} (default), returns quantile values;
#' if \code{FALSE}, returns case depth values.
#' @param ... Additional arguments passed to internal methods.
#' @author
#' 
#' Min Lu and Hemant Ishwaran
#' @seealso \command{\link{isopro}} \command{\link{uvarpro}}
#' \command{\link{varpro}}
#' @references
#' 
#' Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. (2008). Isolation forest.
#' 2008 Eighth IEEE International Conference on Data Mining. IEEE.
#' 
#' Ishwaran H. (2025).  Multivariate Statistics: Classical Foundations and
#' Modern Machine Learning, CRC (Chapman and Hall), in press.
#' @keywords predict outlier
#' @examples
#' 
#' \donttest{
#' ## ------------------------------------------------------------
#' ##
#' ## boston housing
#' ## unsupervised isopro analysis
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## training
#' data(BostonHousing, package = "mlbench")
#' o <- isopro(data=BostonHousing)
#' 
#' ## make fake data
#' fake <- do.call(rbind, lapply(1:nrow(BostonHousing), function(i) {
#'   fakei <- BostonHousing[i,]
#'   fakei$lstat <- quantile(BostonHousing$lstat, .99)
#'   fakei$nox <- quantile(BostonHousing$nox, .99)
#'   fakei
#' }))
#' 
#' ## compare depth values for fake data to training data
#' depth.fake <- predict(o, fake)
#' depth.train <- predict(o)
#' depth.data <- rbind(data.frame(whichdata="fake", depth=depth.fake),
#'                     data.frame(whichdata="train", depth=depth.train))
#' boxplot(depth~whichdata, depth.data, xlab="data", ylab="depth quantiles")
#' 
#' 
#' ## ------------------------------------------------------------
#' ##
#' ## boston housing
#' ## isopro supervised analysis with different split rules
#' ##
#' ## ------------------------------------------------------------
#' 
#' data(BostonHousing, package="mlbench")
#' 
#' ## supervised isopro analysis using different splitrules
#' o <- isopro(formula=medv~.,data=BostonHousing)
#' o.hvwt <- isopro(formula=medv~.,data=BostonHousing,splitrule="mse.hvwt")
#' o.unwt <- isopro(formula=medv~.,data=BostonHousing,splitrule="mse.unwt")
#'      
#' ## make fake data
#' fake <- do.call(rbind, lapply(1:nrow(BostonHousing), function(i) {
#'   fakei <- BostonHousing[i,]
#'   fakei$lstat <- quantile(BostonHousing$lstat, .99)
#'   fakei$nox <- quantile(BostonHousing$nox, .99)
#'   fakei
#' }))
#' 
#' ## compare depth values for fake data to training data
#' depth.train <- predict(o)
#' depth.hvwt.train <- predict(o.hvwt)
#' depth.unwt.train <- predict(o.unwt)
#' depth.fake <- predict(o, fake)
#' depth.hvwt.fake <- predict(o.hvwt, fake)
#' depth.unwt.fake <- predict(o.unwt, fake)
#' depth.data <- rbind(data.frame(whichdata="fake", depth=depth.fake),
#'                     data.frame(whichdata="fake.hvwt", depth=depth.hvwt.fake),
#'                     data.frame(whichdata="fake.unwt", depth=depth.unwt.fake),
#'                     data.frame(whichdata="train", depth=depth.train),
#'                     data.frame(whichdata="train.hvwt", depth=depth.hvwt.train),
#'                     data.frame(whichdata="train.unwt", depth=depth.unwt.train))
#' boxplot(depth~whichdata, depth.data, xlab="data", ylab="depth quantiles")
#' 
#' }
#' 
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
