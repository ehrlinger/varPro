#' Prediction on Test Data using Unsupervised VarPro
#' 
#' Obtain predicted values on test data for unsupervised forests.
#' 
#' 
#' Applies to unsupervised VarPro objects built using the autoencoder
#' (\code{method = "auto"}). The object contains a multivariate random forest
#' used to generate predictions for the test data.
#' 
#' Returns a matrix of predicted values, where each column corresponds to a
#' feature (with one-hot encoding applied). The result includes the following
#' attributes:
#' 
#' \enumerate{
#' 
#' \item \code{mse}: Standardized mean squared error averaged across features.
#' 
#' \item \code{mse.all}: Standardized mean squared error for each individual
#' feature.
#' 
#' }
#' 
#' @param object Unsupervised VarPro object from a previous call to
#' \code{uvarpro}. Only applies if \code{method = "auto"} was used.
#' @param newdata Optional test data. If not provided, the training data is
#' used.
#' @param ... Additional arguments passed to internal methods.
#' @author
#' 
#' Min Lu and Hemant Ishwaran
#' @seealso \command{\link{uvarpro}}
#' @keywords predict uvarpro
#' @examples
#' 
#' \donttest{
#' ## ------------------------------------------------------------
#' ##
#' ## boston housing
#' ## obtain predicted values for the training data
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## unsupervised varpro on boston housing
#' data(BostonHousing, package = "mlbench")
#' o <- uvarpro(data=BostonHousing)
#' 
#' ## predicted values for the training features
#' print(head(predict(o)))
#' 
#' ## ------------------------------------------------------------
#' ##
#' ## mtcars
#' ## obtain predicted values for test data
#' ## also illustrates hot-encoding working on test data
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## mtcars with some factors
#' d <- data.frame(mpg=mtcars$mpg,lapply(mtcars[, c("cyl", "vs", "carb")], as.factor))
#' 
#' ## training 
#' o <- uvarpro(d[1:20,])
#' 
#' ## predicted values on test data
#' print(predict(o, d[-(1:20),]))
#' 
#' ## predicted values on bad test data with strange factor values 
#' dbad <- d[-(1:20),]
#' dbad$carb <- as.character(dbad$carb)
#' dbad$carb <-  sample(LETTERS, size = nrow(dbad))
#' print(predict(o, dbad))
#' 
#' }
#' 
predict.uvarpro <- function(object, newdata, ...) {
  ## check coherence: failure is fatal
  if (!inherits(object, "uvarpro")) {
    stop("object must be an 'uvarpro' varpro object")
  }
  if (object$rf$family != "regr+") {
    stop("only applies to unsupervised varpro objects using auto-encoder")
  }
  ## if test data missing revert to original data
  if (missing(newdata)) {
    newdata <- object$x
  }
  ## otherwise hot-encode it
  else {
    newdata <- get.hotencode.test(object$x, newdata)
  }
  ## predict on newdata (use training data otherwise)
  oo <- predict.rfsrc(object$rf, newdata)
  xhat <- get.mv.predicted(oo, oob = TRUE)
  colnames(xhat) <- oo$xvar.names
  ## standardized mse values
  mse.all <- colMeans((xhat - oo$xvar)^2, na.rm = TRUE) / apply(oo$xvar, 2, var, na.rm = TRUE)
  mse.all[is.infinite(mse.all)] <- NA
  mse <- mean(mse.all, na.rm = TRUE)
  ## return the goodies
  attr(xhat, "mse") <- mse
  attr(xhat, "mse.all") <- mse.all
  xhat
}
