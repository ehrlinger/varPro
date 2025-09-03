#' Unsupervised Variable Selection using Variable Priority (UVarPro)
#' 
#' Variable Selection in Unsupervised Problems using UVarPro
#' 
#' 
#' UVarPro performs unsupervised variable selection by applying the VarPro
#' framework to random forests trained on unlabeled data. The forest
#' construction is governed by the \code{method} argument. By default,
#' \code{method = "auto"} fits a random forest autoencoder, which regresses
#' each selected variable on itself, a specialized form of multivariate forest
#' modeling. Alternatives include \code{"unsupv"}, which uses pseudo-responses
#' and multivariate splits to build an unsupervised forest (Tang and Ishwaran,
#' 2017), and \code{"rnd"}, which uses completely random splits. For large
#' datasets, the autoencoder may be slower, while the \code{"unsupv"} and
#' \code{"rnd"} options are typically more computationally efficient.
#' 
#' Variable importance is measured using an entropy-based criterion that
#' reflects the overall variance explained by each feature. Users may also
#' supply custom entropy functions to define alternative importance metrics.
#' See the examples for details.
#' 
#' @param data Data frame containing the unsupervised data.
#' @param method Type of forest used. Options are \code{"auto"} (auto-encoder),
#' \code{"unsupv"} (unsupervised analysis), and \code{"rnd"} (pure random
#' forest).
#' @param ntree Number of trees to grow.
#' @param nodesize Minimum terminal node size. If not specified, an internal
#' function selects an appropriate value based on sample size and dimension.
#' @param max.rules.tree Maximum number of rules per tree.
#' @param max.tree Maximum number of trees used to extract rules.
#' @param papply Parallel apply method; typically \code{mclapply} or
#' \code{lapply}.
#' @param verbose Print verbose output?
#' @param seed Seed for reproducibility.
#' @param ... Additional arguments passed to \code{rfsrc}.
#' @return
#' 
#' A uvarpro object.
#' @author Min Lu and Hemant Ishwaran
#' @seealso \command{\link{varpro}}
#' @references
#' 
#' Tang F. and Ishwaran H. (2017).  Random forest missing data algorithms.
#' \emph{Statistical Analysis and Data Mining}, 10:363-377.
#' @keywords uvarpro
#' @examples
#' 
#' ## ------------------------------------------------------------
#' ## boston housing: default call
#' ## ------------------------------------------------------------
#' 
#' data(BostonHousing, package = "mlbench")
#' 
#' ## default call
#' o <- uvarpro(BostonHousing)
#' print(importance(o))
#' 
#' ## ------------------------------------------------------------
#' ## boston housing: using method="unsupv"
#' ## ------------------------------------------------------------
#' 
#' data(BostonHousing, package = "mlbench")
#' 
#' ## unsupervised splitting 
#' o <- uvarpro(BostonHousing, method = "unsupv")
#' print(importance(o))
#' 
#' \donttest{
#' 
#' ## ------------------------------------------------------------
#' ## boston housing: illustrates hot-encoding
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
#' ## call unsupervised varpro and print importance
#' print(importance(o <- uvarpro(Boston)))
#' 
#' ## get top variables
#' get.topvars(o)
#' 
#' ## map importance values back to original features
#' print(get.orgvimp(o))
#' 
#' ## same as above ... but for all variables
#' print(get.orgvimp(o, pretty = FALSE))
#' 
#' 
#' ## ------------------------------------------------------------
#' ## latent variable simulation
#' ## ------------------------------------------------------------
#' 
#' n <- 1000
#' w <- rnorm(n)
#' x <- rnorm(n)
#' y <- rnorm(n)
#' z <- rnorm(n)
#' ei <- matrix(rnorm(n * 20, sd = sqrt(.1)), ncol = 20)
#' e21 <- rnorm(n, sd = sqrt(.4))
#' e22 <- rnorm(n, sd = sqrt(.4))
#' wi <- w + ei[, 1:5]
#' xi <- x + ei[, 6:10]
#' yi <- y + ei[, 11:15]
#' zi <- z + ei[, 16:20]
#' h1 <- w + x + e21
#' h2 <- y + z + e22
#' dta <- data.frame(w=w,wi=wi,x=x,xi=xi,y=y,yi=yi,z=z,zi=zi,h1=h1,h2=h2)
#' 
#' ## default call
#' print(importance(uvarpro(dta)))
#' 
#' 
#' ## ------------------------------------------------------------
#' ## glass (remove outcome)
#' ## ------------------------------------------------------------
#' 
#' data(Glass, package = "mlbench")
#' 
#' ## remove the outcome
#' Glass$Type <- NULL
#' 
#' ## get importance
#' o <- uvarpro(Glass)
#' print(importance(o))
#' 
#' ## compare to PCA
#' (biplot(prcomp(o$x, scale = TRUE)))
#' 
#' ## ------------------------------------------------------------
#' ## largish data set: illustrates various options to speed up calculations
#' ## ------------------------------------------------------------
#' 
#' ## first we roughly impute the data
#' data(housing, package = "randomForestSRC")
#' 
#' ## to speed up analysis, convert all factors to real values
#' housing2 <- randomForestSRC:::get.na.roughfix(housing)
#' housing2 <- data.frame(data.matrix(housing2))
#' 
#' ## use fewer trees and bigger nodesize
#' print(importance(uvarpro(housing2, ntree = 50, nodesize = 150)))
#' 
#' ## ------------------------------------------------------------
#' ##  custom importance
#' ##  OPTION 1: use hidden entropy option
#' ## ------------------------------------------------------------
#' 
#' my.entropy <- function(xC, xO, ...) {
#' 
#'   ## xC     x feature data from complementary region
#'   ## xO     x feature data from original region
#'   ## ...    used to pass aditional options (required)
#'  
#'   ## custom importance value
#'   wss <- mean(apply(rbind(xO, xC), 2, sd, na.rm = TRUE))
#'   bss <- (mean(apply(xC, 2, sd, na.rm = TRUE)) +
#'               mean(apply(xO, 2, sd, na.rm = TRUE)))
#'   imp <- 0.5 * bss / wss
#'   
#'   ## entropy value must contain complementary and original membership
#'   entropy <- list(comp = list(...)$compMembership,
#'                   oob = list(...)$oobMembership)
#' 
#'   ## return importance and in the second slot the entropy list 
#'   list(imp = imp, entropy)
#' 
#' }
#' 
#' o <- uvarpro(BostonHousing, entropy=my.entropy)
#' print(importance(o))
#' 
#' 
#' ## ------------------------------------------------------------
#' ##  custom importance
#' ##  OPTION 2: direct importance without hidden entropy option
#' ## ------------------------------------------------------------
#' 
#' o <- uvarpro(BostonHousing, ntree=3, max.rules.tree=10)
#' 
#' ## convert original/release region into two-class problem
#' ## define importance as the lasso beta values 
#' 
#' ## For faster performance on Unix systems, consider using:
#' ## library(parallel)
#' ## imp <- do.call(rbind, mclapply(seq_along(o$entropy), function(j) { ... }))
#' 
#' imp <- do.call(rbind, lapply(seq_along(o$entropy), function(j) {
#'   rO <- do.call(rbind, lapply(o$entropy[[j]], function(r) {
#'     xC <- o$x[r[[1]],names(o$entropy),drop=FALSE]
#'     xO <- o$x[r[[2]],names(o$entropy),drop=FALSE]
#'     y <- factor(c(rep(0, nrow(xC)), rep(1, nrow(xO))))
#'     x <- rbind(xC, xO)
#'     x <- x[, colnames(x) != names(o$entropy)[j]]
#'     fit <- tryCatch(
#'       suppressWarnings(glmnet::cv.glmnet(as.matrix(x), y, family = "binomial")),
#'       error = function(e) NULL
#'     )
#'     if (!is.null(fit)) {
#'       beta <- setNames(rep(0, length(o$entropy)), names(o$entropy))
#'       bhat <- abs(coef(fit)[-1, 1])
#'       beta[names(bhat)] <- bhat
#'       beta
#'     } else {
#'       NULL
#'     }
#'   }))
#'   if (!is.null(rO)) {
#'     val <- colMeans(rO, na.rm = TRUE)
#'     names(val) <- colnames(rO)
#'     return(val)
#'   } else {
#'     return(NULL)
#'   }
#' }) |> setNames(names(o$entropy)))
#' 
#' print(imp)
#' 
#' 
#' ## ------------------------------------------------------------
#' ##  custom importance
#' ##  OPTION 3: direct importance using built in lasso beta function
#' ## ------------------------------------------------------------
#' 
#' o <- uvarpro(BostonHousing, ntree=3, max.rules.tree=10)
#' print((beta <- get.beta.entropy(o)))
#' 
#' ## bonus: display s-dependent graph
#' sdependent(beta)
#' 
#' }
uvarpro <- function(data,
                    method = c("auto", "unsupv", "rnd"),
                    ntree = 200, nodesize = NULL,
                    max.rules.tree = 20, max.tree = 200,
                    papply = mclapply, verbose = FALSE, seed = NULL,
                    ...)
{		   
  ##------------------------------------------------------------------
  ##
  ##
  ## pre-processing 
  ##
  ##
  ##------------------------------------------------------------------
  # set method
  method <- match.arg(method, c("auto", "unsupv", "rnd"))
  ## data must be a data frame without missing values
  data <- data.frame(na.omit(data))
  ## droplevels
  data <- droplevels(data)
  ## initialize the seed
  seed <- get.seed(seed)
  ##--------------------------------------------------------------
  ##
  ## define the entropy function (or obtain user specified one)
  ##
  ## can be a number or a list
  ## for numbers: this is the importance
  ## for lists: entry 1 = importance; entry 2 = entropy values
  ## 
  ##
  ##--------------------------------------------------------------
  dots <- list(...)
  custom.entropy.flag <- FALSE
  ## default entropy returns a list
  ## first entry = total variance
  ## second entry = pc-simple results
  if (is.null(dots$entropy)) {
    entropy.function <- entropy.default
    get.entropy <- get.entropy.default
  }
  ## user specified entropy function
  else {
    custom.entropy.flag <- TRUE
    entropy.function <- dots$entropy
  }
  ##--------------------------------------------------------------
  ##
  ## extract additional options specified by user
  ## we lock this down to allowed types
  ## define the entropy function used for importance
  ##
  ##--------------------------------------------------------------
  enames <- names(formals(entropy.function))[-(1:2)]
  enames <- setdiff(enames, "...")
  dots.entropy <- dots[names(dots) %in% enames]
  diffnames <- setdiff(enames, names(dots.entropy))
  if (length(diffnames) > 0) {
    dots.entropy <- append(dots.entropy, formals(entropy.function)[diffnames])
  }
  user.provided.varpro.flag <- FALSE
  ## special feature allowing user to pass in an arbitrary varpro object
  ## the purpose of this is to allow access to the entropy function framework
  if (!is.null(dots$object)) {
    user.provided.varpro.flag <- TRUE
    o <- dots$object
    ## over-ride the supplied data if this is a varpro object
    if (inherits(o, "varpro")) {
      data <- o$x[, o$xvar.names, drop = FALSE]
    }
  }
  ## list of (non-hidden) forest parameters
  rfnames <- names(formals(rfsrc))
  ## restrict to allowed values
  rfnames <- rfnames[rfnames != "formula" &
                     rfnames != "data" &
                     rfnames != "ntree" &
                     rfnames != "nodesize" &
                     rfnames != "perf.type"]
  ## get the permissible hidden options for rfrsc
  dots <- dots[names(dots) %in% rfnames]
  ##-----------------------------------------------------------------
  ##
  ## process data
  ##
  ##
  ##------------------------------------------------------------------
  ## remove any column with less than two unique values
  #void.var <- sapply(data, function(x){length(unique(x, na.rm = TRUE)) < 2})
  #if (sum(void.var) > 0) {
  #  data[, which(void.var)] <- NULL
  #}
  ## save the original names
  xvar.org.names <- colnames(data)
  ## hot encode the data
  data <- get.hotencode(data, papply)
  ## assign the xvar names
  xvar.names <- colnames(data)
  ##------------------------------------------------------------------
  ##
  ##
  ## unsupervised forests
  ##
  ##
  ##------------------------------------------------------------------
  if (method == "unsupv" && !user.provided.varpro.flag) {
    dots$ytry <- set.unsupervised.ytry(nrow(data), ncol(data), dots$ytry)
    o <- do.call("rfsrc", c(list(
                   data = data,
                   ntree = ntree,
                   nodesize = set.unsupervised.nodesize(nrow(data), ncol(data), nodesize),
                   perf.type = "none"), dots))
  }
  ##------------------------------------------------------------------
  ##
  ##
  ## pure random forests
  ##
  ##
  ##------------------------------------------------------------------
  if (method == "rnd" && !user.provided.varpro.flag) {
    dots$splitrule <- NULL
    o <- do.call("rfsrc", c(list(formula = yxyz123~.,
                   data = data.frame(yxyz123 = rnorm(nrow(data)), data),
                   splitrule = "random",
                   ntree = ntree,
                   nodesize = set.unsupervised.nodesize(nrow(data), ncol(data) + 1, nodesize),
                   perf.type = "none"), dots))
  }
  ##------------------------------------------------------------------
  ##
  ##
  ## auto-encoder (regr+)
  ##
  ##
  ##------------------------------------------------------------------
  if (method == "auto" && !user.provided.varpro.flag) {
    ## call regr+
    o <- do.call("rfsrc", c(list(formula = get.mv.formula(paste0("y.", xvar.names)),
                   data = data.frame(y = data, data),
                   ntree = ntree,
                   nodesize = set.unsupervised.nodesize(nrow(data), ncol(data), nodesize),
                   perf.type = "none"), dots))
  }
  ##------------------------------------------------------------------
  ##
  ##
  ## call varpro.strength and extract necessary information
  ##
  ##
  ##------------------------------------------------------------------
  ## switch for varpro strength depends on whether object is a forest 
  oo <- get.varpro.strength(o, membership = TRUE, max.rules.tree = max.rules.tree, max.tree = max.tree)
  ## identify useful rules and variables at play
  keep.rules <- which(oo$strengthArray$oobCT > 0 & oo$strengthArray$compCT > 0)
  ## membership lists 
  oobMembership <- oo$oobMembership
  compMembership <- oo$compMembership
  ## keep track of which variable is released for a rule
  xreleaseId <- oo$strengthArray$xReleaseID
  ## standardize x
  x <- scaleM(data, center = FALSE)
  ## used to store the new importance values
  results <- oo$strengthArray[, 1:5, drop = FALSE]
  colnames(results) <- c("tree", "branch", "variable", "n.oob", "imp")
  results$imp <- NA
  ##------------------------------------------------------------------
  ##
  ##
  ## obtain the "X" importance values
  ## - uses the default entropy function (can be user specified) 
  ## - data is ordered so that first coordinate is the target variable
  ##   potentially this allows refined/customization of the entropy function 
  ##
  ##------------------------------------------------------------------
  ## add some useful information for the entropy function
  dots.entropy$xvar.names <- xvar.names
  dots.entropy$data <- data
  ## set the dimension
  p <- ncol(x)
  ## extract entropy
  if (length(keep.rules) > 0) {
    impO <- papply(keep.rules, function(i) {
      ordernms <- c(xreleaseId[i], setdiff(1:p, xreleaseId[i]))
      dots.entropy$oobMembership <- oobMembership[[i]]
      dots.entropy$compMembership <- compMembership[[i]]
      xO <- x[oobMembership[[i]], ordernms, drop = FALSE]
      xC <- x[compMembership[[i]], ordernms, drop = FALSE]
      val <- do.call("entropy.function", c(list(xC, xO), dots.entropy))
      if (!is.list(val)) {
        list(imp = val, attr = NULL, xvar = xreleaseId[i])
      }
      else {
        list(imp = val[[1]], attr = val[[2]], xvar = xreleaseId[i])
      }
    })
    ## extract importance
    imp <- unlist(lapply(impO, "[[", 1))
    results$imp[keep.rules] <- imp
    ## extract entropy values -- this allows for customization via the second slot
    entropy.values <- lapply(impO, "[[", 2)
    if (length(!sapply(entropy.values, is.null)) == 0) {
      entropy.values <- NULL
    }
    else {
      xreleaseId <- unlist(lapply(impO, "[[", 3))
      xreleaseIdUnq <- sort(unique(xreleaseId))
      entropy.values <- lapply(xreleaseIdUnq, function(k) {
        ii <- entropy.values[xreleaseId == k]
        ii[!sapply(ii, is.null)]        
      })
      names(entropy.values) <- xvar.names[xreleaseIdUnq]
    }
  }
  ## no viable rules
  else {
    entropy.values <- NULL
  }
  ##------------------------------------------------------------------
  ##
  ##
  ## gets default entropy values and packages them up nicely
  ##
  ##
  ##------------------------------------------------------------------
  if (!custom.entropy.flag) {
    getnames <- names(formals(get.entropy))[-(1:2)]
    getnames <- setdiff(getnames, "...")
    dots.get <- dots[names(dots) %in% getnames]
    diffnames <- setdiff(getnames, names(dots.get))
    if (length(diffnames) > 0) {
      dots.get <- append(dots.get, formals(get.entropy)[diffnames])
    }
    entropy.values <- do.call("get.entropy",
                           c(list(entropy.values, xvar.names), dots.get))
  }
  ##------------------------------------------------------------------
  ##
  ##
  ## package results up as a varpro object
  ##
  ##
  ##------------------------------------------------------------------
  rO <- list()
  rO$rf <- o
  rO$results <- results
  rO$x <- data
  rO$xvar.names <- xvar.names
  rO$xvar.org.names <- xvar.org.names
  rO$y <- NULL
  rO$y.org <- NULL
  rO$xvar.wt <- rep(1, length(xvar.names))
  rO$max.rules.tree <- max.rules.tree
  rO$max.tree <- max.tree
  rO$entropy <- entropy.values
  rO$family <- "unsupv"
  class(rO) <- "uvarpro"
  rO
}
