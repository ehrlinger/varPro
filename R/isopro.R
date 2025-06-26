#' Identify Anomalous Data
#' 
#' Use isolation forests to identify rare/anomalous data.
#' 
#' 
#' Isolation Forest (Liu et al., 2008) is a random forest-based method for
#' detecting anomalous observations. In its original form, trees are
#' constructed using pure random splits, with each tree built from a small
#' subsample of the data, typically much smaller than the standard 0.632
#' fraction used in random forests. The idea is that anomalous or rare
#' observations are more likely to be isolated early, requiring fewer splits to
#' reach terminal nodes. Thus, observations with relatively small depth values
#' (i.e., shallow nodes) are considered anomalies.
#' 
#' There are several ways to apply the method:
#' 
#' \itemize{
#' 
#' \item The default approach is to supply a \code{formula} and \code{data} to
#' build a supervised isolation forest. If only \code{data} is provided (i.e.,
#' no response), an unsupervised analysis is performed. In this case, the
#' \code{method} option is used to specify the type of isolation forest (e.g.,
#' \code{"unsupv"}, \code{"rnd"}, or \code{"auto"}).
#' 
#' \item If both a formula and data are provided, a supervised model is fit. In
#' this case, \code{method} is ignored. While less conventional, this approach
#' may be useful in certain applications.
#' 
#' \item Alternatively, a \code{varpro} object may be supplied, but other
#' configurations are also supported. In this setting, isolation forest is
#' applied to the reduced feature matrix extracted from theobject. This is
#' similar to using the \code{data} option alone but with the advantage of
#' prior dimension reduction.
#' 
#' }
#' 
#' Users are encouraged to experiment with the choice of \code{method}, as the
#' original isolation forest (\code{"rnd"}) performs well in many scenarios but
#' can be improved upon in others. For example, in some cases, \code{"unsupv"}
#' or \code{"auto"} may yield better detection performance.
#' 
#' In terms of computational cost, \code{"rnd"} is the fastest, followed by
#' \code{"unsupv"}. The slowest is \code{"auto"}, which is best suited for
#' low-dimensional settings.
#' 
#' @param object \code{varpro} object returned from a previous call.
#' @param method Isolation forest method. Options are \code{"unsupv"}
#' (unsupervised analysis, default), \code{"rnd"} (pure random splitting), and
#' \code{"auto"} (auto-encoder, a type of multivariate forest).
#' @param sampsize Function or numeric value specifying the sample size used
#' for constructing each tree. Sampling is without replacement.
#' @param ntree Number of trees to grow.
#' @param nodesize Minimum terminal node size.
#' @param formula Formula used for supervised isolation forest. Ignored if
#' \code{object} is provided.
#' @param data Data frame used to fit the isolation forest. Ignored if
#' \code{object} is provided.
#' @param ... Additional arguments passed to \code{rfsrc}.
#' @author
#' 
#' Min Lu and Hemant Ishwaran
#' @seealso \command{\link{predict.isopro}} \command{\link{uvarpro}}
#' \command{\link{varpro}}
#' @references
#' 
#' Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. (2008). Isolation forest.
#' 2008 Eighth IEEE International Conference on Data Mining. IEEE.
#' 
#' Ishwaran H. (2025).  Multivariate Statistics: Classical Foundations and
#' Modern Machine Learning, CRC (Chapman and Hall), in press.
#' @keywords outlier
#' @examples
#' 
#' 
#' ## ------------------------------------------------------------
#' ##
#' ## satellite data: convert some of the classes to "outliers"
#' ## unsupervised isopro analysis
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## load data, make three of the classes into outliers
#' data(Satellite, package = "mlbench")
#' is.outlier <- is.element(Satellite$classes,
#'           c("damp grey soil", "cotton crop", "vegetation stubble"))
#' 
#' ## remove class labels, make unsupervised data
#' x <- Satellite[, names(Satellite)[names(Satellite) != "classes"]]
#' 
#' ## isopro calls
#' i.rnd <- isopro(data=x, method = "rnd", sampsize=32)
#' i.uns <- isopro(data=x, method = "unsupv", sampsize=32)
#' i.aut <- isopro(data=x, method = "auto", sampsize=32)
#' 
#' ## AUC and precision recall (computed using true class label information)
#' perf <- cbind(get.iso.performance(is.outlier,i.rnd$howbad),
#'               get.iso.performance(is.outlier,i.uns$howbad),
#'               get.iso.performance(is.outlier,i.aut$howbad))
#' colnames(perf) <- c("rnd", "unsupv", "auto")
#' print(perf)
#' 
#' \donttest{
#' ## ------------------------------------------------------------
#' ##
#' ## boston housing analysis
#' ## isopro analysis using a previous VarPro (supervised) object 
#' ##
#' ## ------------------------------------------------------------
#' 
#' data(BostonHousing, package = "mlbench")
#' 
#' ## call varpro first and then isopro
#' o <- varpro(medv~., BostonHousing)
#' o.iso <- isopro(o)
#' 
#' ## identify data with extreme percentiles
#' print(BostonHousing[o.iso$howbad <= quantile(o.iso$howbad, .01),])
#' 
#' ## ------------------------------------------------------------
#' ##
#' ## boston housing analysis
#' ## supervised isopro analysis - direct call using formula/data
#' ##
#' ## ------------------------------------------------------------
#' 
#' data(BostonHousing, package = "mlbench")
#' 
#' ## direct approach uses formula and data options
#' o.iso <- isopro(formula=medv~., data=BostonHousing)
#' 
#' ## identify data with extreme percentiles
#' print(BostonHousing[o.iso$howbad <= quantile(o.iso$howbad, .01),])
#' 
#' 
#' ## ------------------------------------------------------------
#' ##
#' ## monte carlo experiment to study different methods
#' ## unsupervised isopro analysis
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## monte carlo parameters
#' nrep <- 25
#' n <- 1000
#' 
#' ## simulation function
#' twodimsim <- function(n=1000) {
#'   cluster1 <- data.frame(
#'     x = rnorm(n, -1, .4),
#'     y = rnorm(n, -1, .2)
#'   )
#'   cluster2 <- data.frame(
#'     x = rnorm(n, +1, .2),
#'     y = rnorm(n, +1, .4)
#'   )
#'   outlier <- data.frame(
#'     x = -1,
#'     y =  1
#'   )
#'   x <- data.frame(rbind(cluster1, cluster2, outlier))
#'   is.outlier <- c(rep(FALSE, 2 * n), TRUE)
#'   list(x=x, is.outlier=is.outlier)
#' }
#' 
#' ## monte carlo loop
#' hbad <- do.call(rbind, lapply(1:nrep, function(b) {
#'   cat("iteration:", b, "\n")
#'   ## draw the data
#'   simO <- twodimsim(n)
#'   x <- simO$x
#'   is.outlier <- simO$is.outlier
#'   ## iso pro calls
#'   i.rnd <- isopro(data=x, method = "rnd")
#'   i.uns <- isopro(data=x, method = "unsupv")
#'   i.aut <- isopro(data=x, method = "auto")
#'   ## save results
#'   c(tail(i.rnd$howbad,1),
#'     tail(i.uns$howbad,1),
#'     tail(i.aut$howbad,1))
#' }))
#' 
#' 
#' ## compare performance
#' colnames(hbad) <- c("rnd", "unsupv", "auto")
#' print(summary(hbad))
#' boxplot(hbad,col="blue",ylab="outlier percentile value")
#' }
#' 
#' 
isopro <- function(object,
                   method = c("unsupv", "rnd", "auto"),
                   sampsize = function(x){min(2^6, .632 * x)},
                   ntree = 500, nodesize = 1,
                   formula = NULL, data = NULL, ...) {
  ## ------------------------------------------------------------------------
  ##
  ## coherence checks: determine if this is a varpro object, or formula/data
  ##
  ## ------------------------------------------------------------------------
  ## must be a varpro object
  if (is.null(formula) && is.null(data) && !inherits(object, "varpro")) {
    stop("object must be a varpro object")
  }
  ## convert data to a data fram
  if (!is.null(data) && !is.data.frame(data)) {
    data <- data.frame(data)
  }
  ## if this is a varpro object use this to filter the data
  no.formula.data.flag <- FALSE
  if (is.null(formula) && is.null(data)) {
    no.formula.data.flag <- TRUE
    topvars <- get.topvars(object)
    data <- object$x[, topvars]
  }
  ## set method
  method <- match.arg(method, c("unsupv", "rnd", "auto"))
  ## coherence check for supervised analysis
  if (!is.null(formula) && !is.null(data) && missing(object)) {
    method <- "supv"
  }
  if (method == "supv" && no.formula.data.flag && missing(object)) {
    stop("supervised method requires formula/data to be provided or a varpro object has to be provided")
  }
  ## obtain family and other details for supervised problems
  if (method == "supv") {
    formula <- as.formula(formula)
    o.stump <- get.stump(formula, data)
    family <- o.stump$family
    yvar.names <- o.stump$yvar.names
  }
  ## ------------------------------------------------------------------------
  ##
  ## special treament for imbalanced classification case
  ##
  ## ------------------------------------------------------------------------
  imbalanced.flag <- FALSE
  if (method == "supv" && get.varpro.hidden(NULL, NULL)$use.rfq) {
    if (family == "class" && length(levels(data[, yvar.names])) == 2) {
      y.frq <- table(data[, yvar.names])
      class.labels <- names(y.frq)
      iratio <- max(y.frq, na.rm = TRUE) / min(y.frq, na.rm = TRUE)
      imbalanced.flag <- iratio > get.varpro.hidden(NULL, NULL)$iratio.threshold
    }
  }
  ##--------------------------------------------------------------
  ##
  ## extract additional options specified by user
  ## we lock this down to allowed types
  ##
  ##--------------------------------------------------------------
  ## list of (non-hidden) forest parameters
  rfnames <- names(formals(rfsrc))
  ## restrict to allowed values
  rfnames <- rfnames[rfnames != "formula" &
                     rfnames != "data" &
                     rfnames != "sampsize" &
                     rfnames != "ntree" &
                     rfnames != "nodesize" &
                     rfnames != "perf.type"]
  ## get the permissible hidden options
  dots <- list(...)
  dots <- dots[names(dots) %in% rfnames]
  ## ------------------------------------------------------------------------
  ##
  ## unsupervised iso forests
  ##
  ## ------------------------------------------------------------------------ 
  if (method == "unsupv") {
    if (is.null(dots$mtry)) {
      dots$ytry <- min(ceiling(sqrt(ncol(data))), ncol(data) - 1)
      dots$mtry <- Inf
    }
    o.iso <- do.call("rfsrc", c(list(data = data,
                   sampsize = sampsize,
                   ntree = ntree,
                   nodesize = nodesize,
                   perf.type = "none"), dots))
  }
  ## ------------------------------------------------------------------------
  ##
  ## random split iso
  ##
  ## ------------------------------------------------------------------------ 
  if (method == "rnd") {
    dots$splitrule <- NULL
    o.iso <- do.call("rfsrc", c(list(formula = yxyz123~.,
                   data = data.frame(yxyz123 = rnorm(nrow(data)), data),
                   splitrule = "random",
                   sampsize = sampsize,
                   ntree = ntree,
                   nodesize = nodesize,
                   perf.type = "none"), dots))
  }
  ## ------------------------------------------------------------------------
  ##
  ## multivariate (auto-encoder) iso
  ##
  ## ------------------------------------------------------------------------ 
  if (method == "auto") {
    o.iso <- do.call("rfsrc", c(list(formula = get.mv.formula(paste0("y.", colnames(data))),
                   data = data.frame(y=data, data),
                   sampsize = sampsize,
                   ntree = ntree,
                   nodesize = nodesize,
                   perf.type = "none"), dots))
  }
  ## ------------------------------------------------------------------------
  ##
  ##   ## supervised  iso
  ##
  ## ------------------------------------------------------------------------ 
  if (method == "supv") {
    ## check if this is imbalanced using default threshold setting
    ## by default brf is used, unless the user over-rides this using "use.brf"
    if (imbalanced.flag) {
      ## gini unweighting is exceptionally slow for imbalanced data - turn this off
      if (is.null(dots$splitrule)) {
      #  dots$splitrule <- "gini.unwt"
      }
      if (is.null(dots$brf) || dots$brf == TRUE) {
        dots$brf <- dots$sampsize <- NULL
        o.iso <- do.call("imbalanced", c(list(formula = formula, data = data,
                           method = "brf",
                           ntree = ntree,
                           nodesize = nodesize,
                           perf.type = "none"), dots))
      }
      else {
        dots$sampsize <- sampsize
        dots$brf <- NULL
        o.iso <- do.call("imbalanced", c(list(formula = formula, data = data,
                           ntree = ntree,
                           nodesize = nodesize,
                           perf.type = "none"), dots))
      }
    }
    ## default setting: for now we turn off unweighted splitting - more analysis required
    else {
      if (is.null(dots$splitrule) && family == "regr") {
        #dots$splitrule <- "mse.unwt"
      }
      if (is.null(dots$splitrule) && family == "class") {
        #dots$splitrule <- "gini.unwt"
      }
      o.iso <- do.call("rfsrc", c(list(formula = formula, data = data,
                       sampsize = sampsize,
                       ntree = ntree,
                       nodesize = nodesize,
                       perf.type = "none"), dots))
    }
  }
  ## ------------------------------------------------------------------------
  ##
  ##
  ## case depth values
  ##
  ##
  ## ------------------------------------------------------------------------
  case.depth <- colMeans(predict.rfsrc(o.iso, data, case.depth = TRUE)$case.depth, na.rm = TRUE)
  cdf <- ecdf(case.depth)
  howbad <- cdf(case.depth)
  ## ------------------------------------------------------------------------
  ##
  ##
  ## return the goodies
  ##
  ##
  ## ------------------------------------------------------------------------
  rO <- list(case.depth = case.depth,
             howbad = howbad,
             cdf = cdf,
             isoforest = o.iso)
  class(rO) <- "isopro"
  rO
}
