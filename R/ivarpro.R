#' Individual Variable Priority (iVarPro)
#' 
#' Individual Variable Priority: A Model-Independent Local Gradient Method for
#' Variable Importance
#' 
#' 
#' Understanding individual-level variable importance is critical in
#' applications where personalized decisions are required. Traditional variable
#' importance methods focus on average (population-level) effects and often
#' fail to capture heterogeneity across individuals. In many real-world
#' problems, it is not sufficient to determine whether a variable is important
#' on average, we must also understand how it affects individual predictions.
#' 
#' The VarPro framework identifies feature-space regions through rule-based
#' splitting and computes importance using only observed data. This avoids
#' biases introduced by permutation or synthetic data, leading to robust,
#' population-level importance estimates. However, VarPro does not directly
#' capture individual-level effects.
#' 
#' To address this limitation, individual variable priority (iVarPro) extends
#' VarPro by estimating the local gradient of each feature, quantifying how
#' small changes in a variable influence an individual's predicted outcome.
#' These gradients serve as natural measures of sensitivity and provide an
#' interpretable notion of individualized importance.
#' 
#' iVarPro leverages the release region concept from VarPro. A region \eqn{R}
#' is first defined using VarPro rules. Since using only data within \eqn{R}
#' often results in insufficient sample size for stable gradient estimation,
#' iVarPro releases \eqn{R} along a coordinate \eqn{s}. This means the
#' constraint on \eqn{s} is removed while all others are held fixed, yielding
#' additional variation specifically in the \eqn{s}-direction, precisely what
#' is needed to compute directional derivatives.
#' 
#' Local gradients are then estimated via linear regression on the expanded
#' region. The parameter \code{cut} controls the amount of constraint
#' relaxation. A value of \code{cut = 1} corresponds to one standard deviation
#' of the release coordinate, calibrated automatically from the data.
#' 
#' The flexibility of this framework makes it suitable for quantifying
#' individual-level importance in regression, classification, and survival
#' settings.
#' 
#' @param object \code{varpro} object from a previous call to \code{varpro}, or
#' a \code{rfsrc} object.
#' @param cut Sequence of \eqn{lambda} values used to relax the constraint
#' region in the local linear regression model. Calibrated so that \code{cut =
#' 1} corresponds to one standard deviation of the release coordinate.
#' @param nmin Minimum number of observations required for fitting a local
#' linear model.
#' @param nmax Maximum number of observations allowed for fitting a local
#' linear model.
#' @param y.external Optional user-supplied response vector to use as the
#' dependent variable in the local linear regression. Must match the dimension
#' and type expected for the outcome family.
#' @param noise.na Logical. If \code{TRUE} (default), gradients for noisy or
#' non-signal variables are set to \code{NA}; if \code{FALSE}, they are set to
#' zero.
#' @param papply Apply method; either \code{mclapply} or \code{lapply}.
#' @param max.rules.tree Optional. Maximum number of rules per tree. If
#' unspecified, the value from the \code{varpro} object is used.
#' @param max.tree Optional. Maximum number of trees used for rule extraction.
#' If unspecified, the value from the \code{varpro} object is used.
#' @author
#' 
#' Min Lu and Hemant Ishwaran
#' @seealso \command{\link{varpro}}
#' @references
#' 
#' Lu, M. and Ishwaran, H. (2025). Individual variable priority: a
#' model-independent local gradient method for variable importance.
#' @keywords individual importance
#' @examples
#' 
#' \donttest{
#' ## ------------------------------------------------------------
#' ##
#' ## synthetic regression example 
#' ##
#' ## ------------------------------------------------------------
#' 
#' ## true regression function
#' true.function <- function(which.simulation) {
#'   if (which.simulation == 1) {
#'     function(x1,x2) {1*(x2<=.25) +
#'       15*x2*(x1<=.5 & x2>.25) + (7*x1+7*x2)*(x1>.5 & x2>.25)}
#'   }
#'   else if (which.simulation == 2) {
#'     function(x1,x2) {r=x1^2+x2^2;5*r*(r<=.5)}
#'   }
#'   else {
#'     function(x1,x2) {6*x1*x2}
#'   }
#' }
#' 
#' ## simulation function
#' simfunction = function(n = 1000, true.function, d = 20, sd = 1) {
#'   d <- max(2, d)
#'   X <- matrix(runif(n * d, 0, 1), ncol = d)
#'   dta <- data.frame(list(x = X, y = true.function(X[, 1], X[, 2]) + rnorm(n, sd = sd)))
#'   colnames(dta)[1:d] <- paste("x", 1:d, sep = "")
#'   dta
#' }
#' 
#' ## iVarPro importance plot
#' ivarpro.plot <- function(dta, release=1, combined.range=TRUE,
#'                      cex=1.0, cex.title=1.0, sc=5.0, gscale=30, title=NULL) {
#'   x1 <- dta[,"x1"]
#'   x2 <- dta[,"x2"]
#'   x1n = expression(x^{(1)})
#'   x2n = expression(x^{(2)})
#'   if (release==1) {
#'     if (is.null(title)) title <- bquote("iVarPro Estimated Gradient " ~ x^{(1)})
#'     cex.pt <- dta[,"Importance.x1"]
#'   }
#'   else {
#'     if (is.null(title)) title <- bquote("iVarPro Estimated Gradient " ~ x^{(2)})
#'     cex.pt <- dta[,"Importance.x2"]
#'   }
#'   if (combined.range) {
#'     cex.pt <- cex.pt / max(dta[, c("Importance.x1", "Importance.x2")],na.rm=TRUE)
#'   }
#'   rng <- range(c(x1,x2))
#'   par(mar=c(4,5,5,1),mgp=c(2.25,1.0,0))
#'   par(bg="white")
#'   gscalev <- gscale
#'   gscale <- paste0("gray",gscale)
#'   plot(x1,x2,xlab=x1n,ylab=x2n,
#'        ylim=rng,xlim=rng,
#'        col = "#FFA500", pch = 19,
#'        cex=(sc*cex.pt),cex.axis=cex,cex.lab=cex,
#'        panel.first = rect(par("usr")[1], par("usr")[3], par("usr")[2], par("usr")[4], 
#'                           col = gscale, border = NA))
#'   abline(a=0,b=1,lty=2,col= if (gscalev<50) "white" else "black")
#'   mtext(title,cex=cex.title,line=.5)
#' }
#' 
#' ## simulate the data
#' which.simulation <- 1
#' df <- simfunction(n = 500, true.function(which.simulation))
#' 
#' ## varpro analysis
#' o <- varpro(y~., df)
#' 
#' ## canonical ivarpro analysis
#' imp1 <- ivarpro(o)
#' 
#' ## ivarpro analysis with custom lambda
#' imp2 <- ivarpro(o, cut = seq(.05, .75, length=21))
#' 
#' ## build data for plotting the results
#' df.imp1 <- data.frame(Importance = imp1, df[,c("x1","x2")])
#' df.imp2 <- data.frame(Importance = imp2, df[,c("x1","x2")])
#' 
#' ## plot the results
#' par(mfrow=c(2,2))
#' ivarpro.plot(df.imp1,1)
#' ivarpro.plot(df.imp1,2)
#' ivarpro.plot(df.imp2,1)
#' ivarpro.plot(df.imp2,2)
#' 
#' }
ivarpro <- function(object,
                    cut = seq(.05, 1, length=21),
                    nmin = 20, nmax = 150,
                    y.external = NULL,
                    noise.na = TRUE,
                    papply = mclapply,
                    max.rules.tree = 150,
                    max.tree = 150)
{
  ## allows both varpro and rfsrc object
  if (!inherits(object, "varpro")) {
    if (sum(inherits(object, c("rfsrc", "grow"), TRUE) == c(1, 2)) != 2) {
      stop("This function only works for objects of class 'varpro' or `(rfsrc, grow)'")
    }
    ## this is a random forest object
    ## !! x cannot contain factors !!
    else {
      y <- data.matrix(object$predicted.oob)
      xvar.names <- object$xvar.names
      x <- object$xvar
      if (any(sapply(x, is.factor))) {
        stop("factors not allowed in x-features ... consider using a varpro object instead of a forest object")
      }
    }
  }
  ## this is a varpro object
  else {
    max.rules.tree <- object$max.rules.tree
    max.tree <- object$max.tree
    y <- data.matrix(object$rf$predicted.oob)
    xvar.names <- object$xvar.names
    x <- object$x[, xvar.names]
  }
  ## overwrite y if y.external is provided
  if (!is.null(y.external)) {
    y <- data.matrix(y.external)
  }
  ## final check on nmax
  nmax <- max(nmin, min(nmax, ncol(x) / 10))
  ## call varpro strength and pull relevant information
  o <- get.varpro.strength(object, membership = TRUE,
            max.rules.tree = max.rules.tree, max.tree = max.tree)
  keep.rules <- which(o$strengthArray$oobCT > 0 & o$strengthArray$compCT > 0)
  oobMembership <- o$oobMembership
  compMembership <- o$compMembership
  xreleaseId <- o$strengthArray$xReleaseID
  xreleaseIdUnq <- sort(unique(xreleaseId))
  results <- o$results
  ## y is the OOB estimator - keep in mind that y can be multivariate
  if (ncol(y) == 1 | (object$family == "class" & ncol(y) == 2)) {
    y <- y[, 1]
  }
  ## build up the results data frame with importance values and other rule information
  rO <- data.frame(do.call(rbind, papply(keep.rules, function(i) {
    xO <- x[oobMembership[[i]],, drop=FALSE]
    xC <- x[compMembership[[i]],, drop=FALSE]
    if (is.matrix(y)) {
      yO <- y[oobMembership[[i]],,drop=FALSE]
      yC <- y[compMembership[[i]],,drop=FALSE]
    }
    else {
      yO <- y[oobMembership[[i]]]
      yC <- y[compMembership[[i]]]
    }
    c(
      tree=results[i,"tree"],
      branch=results[i,"branch"],
      variable=xreleaseId[i],
      n.oob=length(oobMembership[[i]]),
      imp=cs.local.importance(yO,yC,xO,xC,idx=xreleaseId[i],cut=cut,noise.na=noise.na,nmin=nmin,nmax=nmax)
      )
  })))
  ## now split the data frame into cases and form the case-specific importance
  csO <- list()
  csO$results <- rO
  csO$xvar.names <- xvar.names
  csO$oobMembership <- oobMembership[keep.rules]
  csO$n <- nrow(x)
  if (is.matrix(y)) {
    lapply(1:ncol(y), function(j) {
      csO$results <- rO[, c((1:4), 4+j)]
      csimp.varpro.workhorse(csO, papply, noise.na)
    })
  }
  else {
    csimp.varpro.workhorse(csO, papply, noise.na)
  }
}
##############################################################
##
## utilities for calculating case-specific importance
##
##############################################################
csimp.varpro.workhorse <- function(o, papply = mclapply, noise.na) {
  xn <- o$xvar.names
  data.frame(do.call(rbind, papply(1:o$n, function(i) {
    pt <- sapply(o$oobMembership, function(l) {is.element(i, l)})
    if (noise.na) {
      imp <- rep(NA, length(xn))
    }
    else {
      imp <- rep(0, length(xn))
    }
    names(imp) <- xn
    if (sum(pt) > 0) {
      df <- o$results[which(pt),,drop=FALSE]
      v <- lapply(split(df, df$variable), function(d) {mean(d$imp, na.rm=TRUE)})
      xidx <- as.numeric(names(v))
      imp[xn[xidx]] <- unlist(v)      
    }
    imp
  })))
}
cs.local.importance <- function(yO, yC, xO, xC, idx, cut, noise.na, nmin, nmax) {
  if (!is.matrix(yC)) {
    grad.est(yO, yC, xO[,idx], xC[,idx], cut, noise.na, nmin, nmax)
  }
  else {
    sapply(1:ncol(yC), function(j) {
      grad.est(yO[,j], yC[,j], xO[,idx], xC[,idx], cut, noise.na, nmin, nmax)
    })
  }
}
grad.est <- function(yO, yC, xO, xC, cut, noise.na, nmin = 10, nmax = 20) {
  mn <- mean(xO, na.rm=TRUE)
  x <- c(xO, xC) - mn
  y <- c(yO, yC)
  grp <- c(rep(1, length(xO)), rep(2, length(xC)))
  sdx <- sd(x, na.rm = TRUE)
  maX(do.call(rbind, lapply(cut, function(scl) {
    pt <- abs(x) <= (sdx * scl)
    if (sum(pt) >= nmin) {
      J <- min(sum(pt), nmax)
      pt <- which(pt)[order(abs(x[pt]))[1:J]]
      x <- scale(x[pt], center = FALSE)
      y <- y[pt]
      o.lm <- tryCatch({suppressWarnings(lm(y~., data.frame(y=y, x=x)))}, error = function(ex) {NULL})
      if (is.null(o.lm)) {
        if (noise.na) c(J, NA, NA) else c(J, 0, .Machine$double.xmax)
      }
      else {
        influence <- x ^ 2 / sum(x ^ 2)
        rsq <- rsq.loo(o.lm$residuals, influence, J)
        c(J, abs(o.lm$coef["x"]), rsq)
      }
    }
    else {
      if (noise.na) c(sum(pt), NA, NA) else c(sum(pt), 0, .Machine$double.xmax)
    }
  })))
}
rsq.loo <- function(r, influence, J, K = 5) {
  hii <- 1 / J + influence
  mean((r / (1 - K * hii) )^2, na.rm = TRUE)
}
maX <- function(df) {
  if (all(is.na(df[, 2]))) {
    NA
  }
  else {
    df[, 2][which.min(df[, 3])]
  }
}
