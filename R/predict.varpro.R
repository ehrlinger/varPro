#' Prediction on Test Data using VarPro
#' 
#' Obtain predicted values on test data for VarPro object.
#' 
#' 
#' VarPro uses rules extracted from a random forest built using guided
#' tree-splitting, where variables are selected based on split-weights computed
#' in a preprocessing step.
#' 
#' This function returns predicted values for the input data. If \code{newdata}
#' is provided, predictions are made on that data; otherwise, out-of-bag
#' predictions for the training data are returned.
#' 
#' @param object VarPro object returned from a previous call to \code{varpro}.
#' @param newdata Optional test data. If not provided, predictions are computed
#' using the training data (out-of-bag).
#' @param ... Additional arguments passed to internal methods.
#' @author
#' 
#' Min Lu and Hemant Ishwaran
#' @seealso \command{\link{varpro}}
#' @references
#' 
#' Lu, M. and Ishwaran, H. (2024). Model-independent variable selection via the
#' rule-based variable priority. arXiv e-prints, pp.arXiv-2409.
#' @keywords predict varpro
#' @examples
#' 
#' \donttest{
#' ## ------------------------------------------------------------
#' ##
#' ## boston housing regression
#' ## obtain predicted values for the training data
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## varpro applied to boston housing data
#' data(BostonHousing, package = "mlbench")
#' o <- varpro(medv~., BostonHousing)
#' 
#' ## predicted values for the training features
#' print(head(predict(o)))
#' 
#' ## ------------------------------------------------------------
#' ##
#' ## iris classification
#' ## obtain predicted values for test data
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## varpro applied to iris data
#' trn <- sample(1:nrow(iris), size = 100, replace = FALSE)
#' o <- varpro(Species~., iris[trn,])
#' 
#' ## predicted values on test data
#' print(data.frame(Species=iris[-trn, "Species"], predict(o, iris[-trn,])))
#' 
#' ## ------------------------------------------------------------
#' ##
#' ## mtcars regression
#' ## obtain predicted values for test data
#' ## also illustrates hot-encoding working on test data
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## mtcars with some factors
#' d <- data.frame(mpg=mtcars$mpg,lapply(mtcars[, c("cyl", "vs", "carb")], as.factor))
#' 
#' ## varpro on training data 
#' o <- varpro(mpg~., d[1:20,])
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
predict.varpro <- function(object, newdata, ...) {
  ## check coherence: failure is fatal
  if (!inherits(object, "varpro")) {
    stop("object must be a varpro object")
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
  get.mv.predicted(predict.rfsrc(object$rf, newdata, ...), oob = TRUE)
}
