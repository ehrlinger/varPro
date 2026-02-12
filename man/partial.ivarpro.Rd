\name{partial.ivarpro}
\alias{partial.ivarpro}
\title{Partial iVarPro Plot for a Single Variable}

\description{
  \code{partial.ivarpro()} draws a base-graphics scatterplot of case-specific
  iVarPro gradients for one predictor: x is the predictor value and y is the
  iVarPro local gradient. Points can be colored (\code{col.var}), sized
  (\code{size.var}), optionally jittered, and optionally overlaid with loess
  smooths stratified by \code{col.var}.
}

\usage{
partial.ivarpro(ivar,
                var,
                col.var = NULL,
                size.var = NULL,
                x = NULL,
                ladder = FALSE,
                ladder.cuts = NULL,
                ladder.max.segments = 3000,
                pch = 16,
                cex = 0.8,
                cex.range = c(0.5, 2),
                main = NULL,
                xlab = NULL,
                ylab = "iVarPro gradient",
                legend = TRUE,
                ...)
}

\arguments{

  \item{ivar}{An object returned by \code{\link{ivarpro}}. If \code{x} is not
    supplied, \code{partial.ivarpro()} uses \code{attr(ivar, "data")} when
    available.}

  \item{var}{Variable to plot (name or column index). Must exist in both
    \code{ivar} and \code{x}.}

  \item{col.var}{Optional variable (column name in \code{x}) used for coloring.
    If treated as categorical, colors are assigned per level; if continuous, a
    color ramp is used and the legend shows selected quantiles.}

  \item{size.var}{Optional variable (column name in \code{x}) used to scale point
    sizes to \code{cex.range}.}

  \item{x}{Optional data.frame or matrix of original feature values.}

  \item{ladder}{Logical. If \code{TRUE}, attempt to draw a ladder-based band
    (vertical segments spanning the range across \code{ladder.cuts}) when path
    membership information is present.}

  \item{ladder.cuts}{Optional subset of ladder indices/values used for the band.}

  \item{ladder.max.segments}{Maximum number of ladder segments to draw.}

  \item{pch}{Point character for the default point style.}

  \item{cex}{Base point size (used when \code{size.var} is not supplied).}

  \item{cex.range}{Min/max point sizes used when \code{size.var} is supplied.}

  \item{main}{Main title (defaults to \code{paste0(var, " vs iVarPro gradient")}).}

  \item{xlab}{X-axis label (defaults to \code{var}).}

  \item{ylab}{Y-axis label (defaults to \code{"iVarPro gradient"}).}

  \item{legend}{Logical. If \code{TRUE} and \code{col.var} is supplied, draw a
    legend describing the color mapping.}

  \item{...}{Additional arguments passed to \code{graphics::plot()}, plus these
    optional controls:

    \describe{
      \item{\code{smooth}}{Logical. Draw loess smooth curves. Default \code{TRUE}.
        For categorical \code{col.var}, one curve per level. For continuous
        \code{col.var}, curves are drawn for strata defined by quantiles.}

      \item{\code{smooth.span}}{Loess span (default \code{0.75}). Other loess
        controls: \code{smooth.degree}, \code{smooth.family}, \code{smooth.lwd},
        \code{smooth.lty}, \code{smooth.alpha}, \code{smooth.min.n},
        \code{smooth.n.grid}.}

      \item{\code{jitter}}{Logical or numeric. Default \code{TRUE}. Adds
        horizontal jitter to x to reduce overplotting. Related options:
        \code{jitter.amount}, \code{jitter.fraction}, \code{jitter.seed}.}

      
      \item{\code{x.dist}}{Character vector controlling an x-axis
        distribution strip. Default \code{"none"}. Supported values
        include \code{"rug"}, \code{"hist"}, \code{"density"}, and
        \code{"auto"} (defaults to \code{c("hist","rug")}). Use
        \code{x.dist = c("hist","rug")} (or \code{c("density","rug")})
        to combine. 

	Related options:
	\code{x.dist.side},
        \code{x.dist.height}, \code{x.dist.pad}, \code{x.dist.col},
        \code{x.dist.border}, \code{x.dist.lwd}, \code{x.dist.lty},
        \code{x.dist.bins}, \code{x.dist.adjust}, \code{x.dist.n},
        \code{x.dist.rug.col}, \code{x.dist.rug.lwd},
        \code{x.dist.rug.ticksize}, \code{x.dist.rug.max}. By default,
        an outline is drawn to keep the strip visible; set
        \code{x.dist.lwd = 0} to suppress the outline.
      }

      \item{\code{col.legend.probs}, \code{col.legend.n}}{Quantiles shown in the
        legend when \code{col.var} is continuous. Default is 5 quantiles:
        \code{c(0.05, 0.25, 0.5, 0.75, 0.95)}.}

      \item{\code{smooth.probs}, \code{smooth.n}}{Quantiles defining strata for
        smooth curves when \code{col.var} is continuous. Default matches the
        legend quantiles.}

      \item{\code{zero.line}}{Logical. Default \code{TRUE}. Adds a dashed
        reference line at gradient 0 (\code{y = 0}). Related options:
        \code{zero.line.col}, \code{zero.line.lty}, \code{zero.line.lwd}.}

      \item{\code{col.var.discrete.max}}{If \code{col.var} is numeric with at most
        this many distinct values, treat it as categorical (default \code{10}).}

      \item{\code{col.style}}{How categorical colors are rendered:
        \code{"auto"} (default), \code{"solid"}, \code{"outline"},
        \code{"binary"}. Related options include \code{col.outline},
        \code{col.binary.pch}, \code{col.dodge}.}
    }
  }
}

\details{
  Jitter is applied for visualization only; loess smooths and the optional
  x-axis distribution strip (\code{x.dist}) are computed using the
  unjittered x-values.
}

\value{
  Invisibly returns \code{TRUE}.
}

\seealso{
  \code{\link{ivarpro}}, \code{\link{varpro}}, \code{\link{shap.ivarpro}}
}

\examples{
\donttest{

## ------------------------------------------------------------
##
## Survival example: peakVO2 partial plot
##
## ------------------------------------------------------------

data(peakVO2, package = "randomForestSRC")

ipv <- ivarpro(varpro(Surv(ttodead, died) ~ ., peakVO2))

## Continuous col.var: legend + smooth strata default to 5 quantiles
partial.ivarpro(ipv, var = "peak.vo2", col.var = "interval", size.var = "y")

## Add an x-axis distribution strip (histogram + rug)
partial.ivarpro(ipv, var = "peak.vo2", col.var = "interval", size.var = "y",
                x.dist = c("hist", "rug"))

## Increase legend/smooth strata (e.g., 7 quantiles)
partial.ivarpro(ipv, var = "peak.vo2", col.var = "interval", size.var = "y",
                col.legend.n = 7, smooth.n = 7)

## Classic 3-quantile view (5\%, 50\%, 95\%)
partial.ivarpro(ipv, var = "peak.vo2", col.var = "interval", size.var = "y",
                col.legend.probs = c(0.05, 0.5, 0.95),
                smooth.probs = c(0.05, 0.5, 0.95))

## Factor col.var example: one smooth per level
partial.ivarpro(ipv, var = "peak.vo2", col.var = "betablok", size.var = "y")

## ------------------------------------------------------------
##
## multiclass example: iris (use target= to choose a class)
##
## ------------------------------------------------------------

data(iris)

vp.ir  <- varpro(Species ~ ., iris, ntree = 50)
ivp.ir <- ivarpro(vp.ir)

## Plot gradients for the "setosa" class (target selects the list element)
partial.ivarpro(ivp.ir, var = "Petal.Length", target = "setosa",
                col.var = "Species", x = iris)

## Alternatively, color by the predicted class probability stored in ivp.ir
partial.ivarpro(ivp.ir, var = "Petal.Length", target = "setosa",
                col.var = "y.setosa")


## ------------------------------------------------------------
##
## multiclass example: wine (advanced example)
##
## ------------------------------------------------------------

data(wine, package = "randomForestSRC")

## Give the class labels nicer names than "3", "4", ..., "9"
wine$quality <- factor(wine$quality)
levels(wine$quality) <- paste0("Q", levels(wine$quality))  # "Q3" "Q4" ... "Q9"

vp  <- varpro(quality ~ ., wine, ntree = 50)
ivp <- ivarpro(vp)

## Available targets correspond to class probability columns
names(ivp)

## Build a plotting data.frame that contains:
##  - predictors (from attr(ivp,"data"))
##  - predicted class probabilities (y.Q3, y.Q4, ..., y.Q9)
##  - the observed class label (quality)
xdat <- attr(ivp, "data")
xdat$quality <- wine$quality

## Plot gradients for the "Q7" class:
## y-axis: d P(Y="Q7" | x) / d alcohol  (iVarPro gradient)
## x-axis: alcohol
## color: observed class label
partial.ivarpro(ivp, var = "alcohol", target = "Q7",
                x = xdat, col.var = "quality")

## Alternatively, color by the model's predicted probability for the same class.
## (These columns come from attr(ivp,"data") as y.<class>)
partial.ivarpro(ivp, var = "alcohol", target = "Q7", col.var = "y.Q7")

}
}

\keyword{hplot}
\keyword{importance}
