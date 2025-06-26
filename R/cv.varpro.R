#' Cross-Validated Cutoff Value for Variable Priority (VarPro)
#' 
#' Selects Cutoff Value for Variable Priority (VarPro).
#' 
#' 
#' Applies VarPro and then selects from a grid of cutoff values the cutoff
#' value for identifying variables that minimizes out-of-sample performance
#' (error rate) of a random forest where the forest is fit to the top variables
#' identified by the given cutoff value.
#' 
#' Additionally, a "conservative" and "liberal" list of variables are returned
#' using a one standard deviation rule.  The conservative list comprises
#' variables using the largest cutoff with error rate within one standard
#' deviation from the optimal cutoff error rate, whereas the liberal list uses
#' the smallest cutoff value with error rate within one standard deviation of
#' the optimal cutoff error rate.
#' 
#' For class imbalanced settings (two class problems where relative frequency
#' of labels is skewed towards one class) the code automatically switches to
#' random forest quantile classification (RFQ; see O'Brien and Ishwaran, 2019)
#' under the gmean (geometric mean) performance metric.
#' 
#' @param f Model formula specifying the outcome and predictors.
#' @param data Training data set (data frame).
#' @param nvar Maximum number of variables to return.
#' @param ntree Number of trees to grow.
#' @param local.std Use locally standardized importance values?
#' @param zcut Grid of positive cutoff values used for selecting top variables.
#' @param nblocks Number of blocks (folds) for cross-validation.
#' @param split.weight Use guided tree-splitting? Variables are selected for
#' splitting with probability proportional to split-weights, obtained by
#' default from a preliminary lasso+tree step.
#' @param split.weight.method Character string or vector specifying how
#' split-weights are generated. Defaults to lasso+tree.
#' @param sparse Use sparse split-weights?
#' @param nodesize Minimum terminal node size. If not specified, an internal
#' function sets the value based on sample size and data dimension.
#' @param max.rules.tree Maximum number of rules per tree.
#' @param max.tree Maximum number of trees used for rule extraction.
#' @param papply Apply method; either \code{mclapply} or \code{lapply}.
#' @param verbose Print verbose output?
#' @param seed Seed for reproducibility.
#' @param fast Use \code{rfsrc.fast} in place of \code{rfsrc}? May improve
#' speed at the cost of accuracy.
#' @param crps Use CRPS (continuous ranked probability score) instead of
#' Harrell's C-index for evaluating survival performance? Applies only to
#' survival families.
#' @param ... Additional arguments passed to \code{varpro}.
#' @return
#' 
#' Output containing importance values for the optimized cutoff value.  A
#' conservative and liberal list of variables is also returned.
#' 
#' Note that importance values are returned in terms of the original features
#' and not their hot-encodings.  For importance in terms of hot-encodings, use
#' the built-in wrapper \command{get.vimp} (see example below).
#' @author Min Lu and Hemant Ishwaran
#' @seealso \command{\link{importance.varpro}} \command{\link{uvarpro}}
#' \command{\link{varpro}}
#' @references
#' 
#' Lu, M. and Ishwaran, H. (2024). Model-independent variable selection via the
#' rule-based variable priority. arXiv e-prints, pp.arXiv-2409.
#' 
#' O'Brien R. and Ishwaran H. (2019).  A random forests quantile classifier for
#' class imbalanced data. \emph{Pattern Recognition}, 90, 232-249.
#' @keywords cv.varpro
#' @examples
#' 
#' \donttest{
#' ## ------------------------------------------------------------
#' ## van de Vijver microarray breast cancer survival data
#' ## high dimensional example
#' ## ------------------------------------------------------------
#'      
#' data(vdv, package = "randomForestSRC")
#' o <- cv.varpro(Surv(Time, Censoring) ~ ., vdv)
#' print(o)
#' 
#' ## ------------------------------------------------------------
#' ## boston housing
#' ## ------------------------------------------------------------
#' 
#' data(BostonHousing, package = "mlbench")
#' print(cv.varpro(medv~., BostonHousing))
#' 
#' ## ------------------------------------------------------------
#' ## boston housing - original/hot-encoded vimp
#' ## ------------------------------------------------------------
#' 
#' ## load the data
#' data(BostonHousing, package = "mlbench")
#' 
#' ## convert some of the features to factors
#' Boston <- BostonHousing
#' Boston$zn <- factor(Boston$zn)
#' Boston$chas <- factor(Boston$chas)
#' Boston$lstat <- factor(round(0.2 * Boston$lstat))
#' Boston$nox <- factor(round(20 * Boston$nox))
#' Boston$rm <- factor(round(Boston$rm))
#' 
#' ## make cv call
#' o <-cv.varpro(medv~., Boston)
#' print(o)
#' 
#' ## importance original variables (default)
#' print(get.orgvimp(o, pretty = FALSE))
#' 
#' ## importance for hot-encoded variables
#' print(get.vimp(o, pretty = FALSE))
#' 
#' ## ------------------------------------------------------------
#' ## multivariate regression example: boston housing
#' ## vimp is collapsed across the outcomes
#' ## ------------------------------------------------------------
#' 
#' data(BostonHousing, package = "mlbench")
#' print(cv.varpro(cbind(lstat, nox) ~., BostonHousing))
#' 
#' ## ------------------------------------------------------------
#' ## iris
#' ## ------------------------------------------------------------
#' 
#' print(cv.varpro(Species~., iris))
#' 
#' ## ------------------------------------------------------------
#' ## friedman 1
#' ## ------------------------------------------------------------
#' 
#' print(cv.varpro(y~., data.frame(mlbench::mlbench.friedman1(1000))))
#' 
#' ##----------------------------------------------------------------
#' ##  class imbalanced problem 
#' ## 
#' ## - simulation example using the caret R-package
#' ## - creates imbalanced data by randomly sampling the class 1 values
#' ## 
#' ##----------------------------------------------------------------
#' 
#' if (library("caret", logical.return = TRUE)) {
#' 
#'   ## experimental settings
#'   n <- 5000
#'   q <- 20
#'   ir <- 6
#'   f <- as.formula(Class ~ .)
#'  
#'   ## simulate the data, create minority class data
#'   d <- twoClassSim(n, linearVars = 15, noiseVars = q)
#'   d$Class <- factor(as.numeric(d$Class) - 1)
#'   idx.0 <- which(d$Class == 0)
#'   idx.1 <- sample(which(d$Class == 1), sum(d$Class == 1) / ir , replace = FALSE)
#'   d <- d[c(idx.0,idx.1),, drop = FALSE]
#'   d <- d[sample(1:nrow(d)), ]
#' 
#'   ## cv.varpro call
#'   print(cv.varpro(f, d))
#' 
#' }
#' 
#' 
#' ## ------------------------------------------------------------
#' ## pbc survival with rmst vector
#' ## note that vimp is collapsed across the rmst values
#' ## similar to mv-regression
#' ## ------------------------------------------------------------
#' 
#' data(pbc, package = "randomForestSRC")
#' print(cv.varpro(Surv(days, status)~., pbc, rmst = c(500, 1000)))
#' 
#' 
#' ## ------------------------------------------------------------
#' ## peak VO2 with cutoff selected using fast option
#' ## (a) C-index (default) (b) CRPS performance metric
#' ## ------------------------------------------------------------
#' 
#' data(peakVO2, package = "randomForestSRC")
#' f <- as.formula(Surv(ttodead, died)~.)
#' 
#' ## Harrel's C-index (default)
#' print(cv.varpro(f, peakVO2, ntree = 100, fast = TRUE))
#' 
#' ## Harrel's C-index with smaller bootstrap
#' print(cv.varpro(f, peakVO2, ntree = 100, fast = TRUE, sampsize = 100))
#' 
#' ## CRPS with smaller bootstrap
#' print(cv.varpro(f, peakVO2, crps = TRUE, ntree = 100, fast = TRUE, sampsize = 100))
#' 
#' ## ------------------------------------------------------------
#' ## largish data set: illustrates various options to speed up calculations
#' ## ------------------------------------------------------------
#' 
#' ## roughly impute the data
#' data(housing, package = "randomForestSRC")
#' housing2 <- randomForestSRC:::get.na.roughfix(housing)
#' 
#' ## use bigger nodesize
#' print(cv.varpro(SalePrice~., housing2, fast = TRUE, ntree = 50, nodesize = 150))
#' 
#' ## use smaller bootstrap
#' print(cv.varpro(SalePrice~., housing2, fast = TRUE, ntree = 50, nodesize = 150, sampsize = 250))
#' 
#' }
#' 
cv.varpro <- function(f, data, nvar = 30, ntree = 150,
                      local.std = TRUE, zcut = seq(0.1, 2, length = 50), nblocks = 10,
                      split.weight = TRUE, split.weight.method = NULL, sparse = TRUE,
                      nodesize = NULL, max.rules.tree = 150, max.tree = min(150, ntree),
                      papply = mclapply, verbose = FALSE, seed = NULL,
                      fast = FALSE, crps = FALSE,
                      ...)
{		   
  ##--------------------------------------------------------------
  ##
  ## extract original yvalue names
  ## re-define the original data in case there are missing values
  ##
  ##--------------------------------------------------------------
  stump <- get.stump(f, data)
  n <- stump$n
  p <- length(stump$xvar.names)
  yvar.names <- stump$yvar.names
  data <- data.frame(stump$yvar, stump$xvar)
  colnames(data)[1:length(yvar.names)] <- yvar.names
  family <- stump$family
  rm(stump)
  ##--------------------------------------------------------------
  ##
  ## extract additional options specified by user
  ##
  ##--------------------------------------------------------------
  dots <- list(...)
  ## set nodesize
  nodesize <- set.cv.nodesize(n, p, nodesize)
  dots$nodesize.reduce <- set.nodesize(n, p, dots$nodesize.reduce)
  dots$nodedepth.reduce <- set.nodedepth.reduce(n, p, dots$nodedepth.reduce)
  if (is.null(dots$sampsize)) {
    dots$nodesize.external <- set.nodesize(n, p, dots$nodesize.external)
  }
  else {
    if (is.function(dots$sampsize)) {
      dots$nodesize.external <- set.nodesize(dots$sampsize(n), p, dots$nodesize.external)
    }
    else {
      dots$nodesize.external <- set.nodesize(dots$sampsize, p, dots$nodesize.external)
    }
  }
  ## set rfq parameters for class imbalanced scenario
  use.rfq <- get.varpro.hidden(NULL, NULL)$use.rfq
  iratio.threshold <- get.varpro.hidden(NULL, NULL)$iratio.threshold
  ##--------------------------------------------------------------
  ##
  ## default settings
  ##
  ##--------------------------------------------------------------
  trn <- 1:n
  newdata <- splitrule <- rfq <- imbalanced.obj <- cens.dist <- NULL
  ##--------------------------------------------------------------
  ##
  ## set the type of sampling, define train/test (fast=TRUE)
  ##
  ##--------------------------------------------------------------
  ## use same inbag/oob members to reduce MC error
  if (!fast) {
    if (is.null(dots$sampsize)) {##default sample size function used by rfsrc.fast
      ssize <- n * .632
    }
    else {
      ssize <- eval(dots$sampsize)
    }
    if (is.function(ssize)) {##user has specified a function
      ssize <- ssize(n)
    }
  }
  ## subsampling is in effect when fast = TRUE
  else {
    ## obtain the requested sample size
    if (is.null(dots$sampsize)) {##default sample size function used by rfsrc.fast
      ssize <- eval(formals(randomForestSRC::rfsrc.fast)$sampsize)
    }
    else {
      ssize <- eval(dots$sampsize)
    }
    if (is.function(ssize)) {##user has specified a function
      ssize <- ssize(n)
    }
    ## now hold out a test data set equal to the tree sample size (if possible)
    if (n > (2 * ssize))  {
      tst <- sample(1:n, size = ssize, replace = FALSE)
      trn <- setdiff(1:n, tst)
      newdata <- data[tst,, drop = FALSE]
    }
  }
  ## custom sample array
  samp <- randomForestSRC:::make.sample(ntree, length(trn), ssize)
  ## pass the sample size to varpro as a hidden option
  dots$sampsize <- ssize
  ##--------------------------------------------------------------
  ##
  ## varpro call
  ##
  ##--------------------------------------------------------------
  o <- do.call("varpro", c(list(f = f, data = data, nvar = nvar, ntree = ntree,
                  split.weight = split.weight, split.weight.method = split.weight.method, sparse = sparse,
                  nodesize = nodesize, max.rules.tree = max.rules.tree, max.tree = max.tree,
		  papply = papply, verbose = verbose, seed = seed), dots))
  ##--------------------------------------------------------------
  ##
  ## extract importance values
  ## map importance values which are hot-encoded back to original data 
  ##
  ##--------------------------------------------------------------
  vorg <- get.orgvimp(o, papply = papply, local.std = local.std)
  xvar.names <- vorg$variable
  imp <- vorg$z
  imp[is.na(imp)] <- 0
  ##--------------------------------------------------------------
  ##
  ## remove zcut values that lead to duplicated models
  ##
  ##--------------------------------------------------------------
  zcut.models <- do.call(rbind, lapply(zcut, function(zz) {
    1 * (imp >= zz)
  }))
  zcut <- zcut[!duplicated(zcut.models)]
  ##--------------------------------------------------------------
  ##
  ## rfq details: only applies to two class imbalanced scenarios
  ##
  ##--------------------------------------------------------------
  if (family == "class" && length(levels(data[, yvar.names])) == 2 && use.rfq) {
    ## calculate imblanced ratio
    y.frq <- table(data[, yvar.names])
    class.labels <- names(y.frq)
    iratio <- max(y.frq, na.rm = TRUE) / min(y.frq, na.rm = TRUE)
    ## check if this is imbalanced using default threshold setting
    if (iratio > iratio.threshold) {
      rfq <- TRUE
      splitrule <- "auc"
      imbalanced.obj <- list(perf.type = "gmean",
                             iratio = iratio,
                             iratio.threshold = iratio.threshold)
    }
  }  
  ##--------------------------------------------------------------
  ##
  ## censoring distribution: only applies to survival families
  ##
  ##--------------------------------------------------------------
  if (family == "surv" && crps) {
    cens.dist <- get.cens.dist(data[trn, c(yvar.names, xvar.names), drop = FALSE],
                        ntree, nodesize, ssize)
  }  
  ##--------------------------------------------------------------
  ##
  ## select zcut using out-of-sample performance
  ##
  ##--------------------------------------------------------------
  ## set the seed
  seed <- get.seed(seed)
  ## loop over zcut sequence and acquire OOB error rate
  err <- do.call(rbind, lapply(zcut, function(zz) {
    pt <- imp >= zz
    if (sum(pt) > 0) {
      if (!fast) {
        err.zz <- get.sderr(rfsrc(f, data[trn, c(yvar.names, xvar.names[pt]), drop = FALSE],
                                  nodesize = nodesize,
                                  ntree = ntree,
                                  rfq = rfq,
                                  splitrule = splitrule,
                                  perf.type = "none",
                                  bootstrap = "by.user",
                                  samp = samp,
                                  seed = seed),
                            nblocks = nblocks,
                            crps = crps,
                            papply = papply,
                            imbalanced.obj = imbalanced.obj,
                            cens.dist = cens.dist)
      }
      else {
        ## nodesize is not deployed because fast subsampling is in play
        err.zz <- get.sderr(randomForestSRC::rfsrc.fast(f, data[trn, c(yvar.names, xvar.names[pt]), drop = FALSE],
                            ntree = ntree,
                            rfq = rfq,
                            splitrule = splitrule,
                            perf.type = "none",
                            forest = TRUE,
                            bootstrap = "by.user",
                            samp = samp,
                            seed = seed),
                         nblocks = nblocks,
                         crps = crps,
                         papply = papply,
                         newdata = newdata,
                         imbalanced.obj = imbalanced.obj,
                         cens.dist = cens.dist)
      }
    }
    else {
      err.zz <- c(NA, NA) 
    }
    if (verbose) {
      cat("zcut value", zz,
          "number variables", sum(pt),
          "error", err.zz[1],
          "sd", err.zz[2], "\n")
    }
    c(zz, sum(pt), err.zz)
  }))
  colnames(err) <- c("zcut", "nvar", "err", "sd")
  ##--------------------------------------------------------------
  ##
  ## return the importance values after filtering 
  ##
  ##--------------------------------------------------------------
  ## minimum error
  vmin <- vorg
  zcut.min <- 0
  if (!all(is.na(err[, 3]))) {
    zcut.min <- zcut[which.min(err[, 3])]
    if (verbose) {
      cat("optimal cutoff value", zcut.min, "\n")
    }
    vmin <- vorg[imp >= zcut.min,, drop = FALSE]
  }
  ## 1sd error rule -conservative
  v1sd.conserve <- vorg
  zcut.1sd <- 0
  if (!all(is.na(err[, 3]))) {
    idx.opt <- which.min(err[, 3])
    serr <- mean(err[, 4], na.rm = TRUE)
    idx2.opt <- err[, 3] < 1 & (err[, 3] <= (err[idx.opt, 3] + serr))
    idx2.opt[is.na(idx2.opt)] <- FALSE
    if (sum(idx2.opt) > 0) {
      zcut.1sd <- zcut[max(which(idx2.opt))]
      if (verbose) {
        cat("optimal 1sd + (conservative) cutoff value", zcut.1sd, "\n")
      }
      v1sd.conserve <- vorg[imp >= zcut.1sd,, drop = FALSE]
    }
    else {
      v1sd.conserve <- NULL
    }
  }
  ## 1sd error rule -liberal
  v1sd.liberal <- vorg
  zcut.liberal <- 0
  if (!all(is.na(err[, 3]))) {
    idx.opt <- which.min(err[, 3])
    serr <- mean(err[, 4], na.rm = TRUE)
    zcut.liberal <- zcut[min(which(err[, 3] <= (err[idx.opt, 3] + serr)), na.rm = TRUE)]
    if (verbose) {
      cat("optimal 1sd - (liberal) cutoff value", zcut.liberal, "\n")
    }
    v1sd.liberal <- vorg[imp >= zcut.liberal,, drop = FALSE]
  }
  rO <- list(imp = vmin,
             imp.conserve = v1sd.conserve,
             imp.liberal = v1sd.liberal,
             err = err,
             zcut = zcut.min,
             zcut.conserve = zcut.1sd,
             zcut.liberal = zcut.liberal)
  class(rO) <- "cv.varpro"
  ## append some useful information as attributes
  attr(rO, "imp.org") <- importance(o, local.std = local.std)
  attr(rO, "xvar.names") <- o$xvar.names
  attr(rO, "xvar.org.names") <- o$xvar.org.names
  attr(rO, "family") <- o$family
  return(rO)
}
## custom print object for cv to make attributes invisible
print.cv.varpro <- function(x, ...) {
  attr(x, "class") <- attr(x, "imp.org") <- attr(x, "xvar.names") <-
    attr(x, "xvar.org.names") <- attr(x, "family") <- NULL
  print(x)
}
print.cv <- print.cv.varpro
