#' Predict using ivarpro paths
#'
#' Applies ivarpro path metadata to produce case-level predicted effects.
#'
#' @param object An ivarpro object.
#' @param newdata Optional data for prediction.
#' @param model Optional model override.
#' @param noise.na Logical; convert unstable values to NA.
#' @param path.store.membership Logical; store membership output.
#' @param save.data Logical; attach processed data to output.
#' @param ... Additional arguments passed to forest prediction.
#'
#' @return A data frame or list depending on the ivarpro target structure.
#' @method predict ivarpro
#' @export
predict.ivarpro <- function(object,
                            newdata = NULL,
                            model = NULL,
                            noise.na = NULL,
                            path.store.membership = FALSE,
                            save.data = TRUE,
                            ...) {
  ## ------------------------------------------------------------
  ## Helpers: resolve common path + model from object (supports compact multivariate output)
  ## ------------------------------------------------------------
  .is_list_out <- function(x) {
    is.list(x) && !inherits(x, "data.frame")
  }
  .has_rule_meta <- function(p) {
    !is.null(p) &&
      !is.null(p$rule.tree) &&
      !is.null(p$rule.branch) &&
      !is.null(p$rule.variable)
  }
  .get_common_path <- function(obj) {
    if (.is_list_out(obj)) {
      ## New compact style: common path stored on the list
      p_common <- attr(obj, "ivarpro.path")
      if (.has_rule_meta(p_common)) return(p_common)
      ## Legacy style: common path stored on each element
      if (length(obj) == 0L) stop("ivarpro object is an empty list.")
      p0 <- attr(obj[[1]], "ivarpro.path")
      if (.has_rule_meta(p0)) return(p0)
      stop("Missing rule metadata in ivarpro.path (rule.tree/branch/variable).")
    } else {
      p0 <- attr(obj, "ivarpro.path")
      if (.has_rule_meta(p0)) return(p0)
      stop("Missing 'ivarpro.path' attribute (object not from current ivarpro()).")
    }
  }
  .get_model <- function(obj) {
    m0 <- attr(obj, "model")
    if (!is.null(m0)) return(m0)
    if (.is_list_out(obj)) {
      m1 <- attr(obj[[1]], "model")
      if (!is.null(m1)) return(m1)
    }
    NULL
  }
  .get_rule_imp <- function(obj_j) {
    p <- attr(obj_j, "ivarpro.path")
    if (!is.null(p) && !is.null(p$rule.imp)) return(p$rule.imp)
    NULL
  }
  ## ------------------------------------------------------------
  ## Resolve common path + model
  ## ------------------------------------------------------------
  path0 <- .get_common_path(object)
  if (is.null(model)) model <- .get_model(object)
  if (is.null(model)) {
    stop("ivarpro prediction requires a model.
",
         "Either store it in ivarpro() (save.model=TRUE) or pass it via predict.ivarpro(..., model=...).")
  }
  ## Noise behavior default: inherit from path if not supplied
  if (is.null(noise.na)) {
    if (!is.null(path0$noise.na)) noise.na <- isTRUE(path0$noise.na) else noise.na <- TRUE
  }
  xnames <- path0$xvar.names
  if (is.null(xnames)) stop("ivarpro.path is missing 'xvar.names'.")
  rule.tree   <- as.integer(path0$rule.tree)
  rule.branch <- as.integer(path0$rule.branch)
  rule.var    <- as.integer(path0$rule.variable)
  R <- length(rule.tree)
  if (R == 0L) stop("No rules found in ivarpro.path (rule.tree length is 0).")
  if (length(rule.branch) != R || length(rule.var) != R) {
    stop("Inconsistent rule metadata lengths in ivarpro.path.")
  }
  ## ------------------------------------------------------------
  ## Prepare rf + (optionally) newx
  ## ------------------------------------------------------------
  rf <- NULL
  if (!is.null(newdata) && !is.data.frame(newdata)) newdata <- as.data.frame(newdata)
  if (inherits(model, "varpro")) {
    if (!is.null(newdata) && isTRUE(attr(model$x, "hotencode"))) {
      xs <- model$x[, xnames, drop=FALSE]
      attr(xs, "hotencode") <- attr(model$x, "hotencode")
      attr(xs, "levels") <- attr(model$x, "levels")
      newdata <- get.hotencode.test(xs, newdata)
    }
    rf <- model$rf
  } else if (inherits(model, "rfsrc") && inherits(model, "grow")) {
    rf <- model
  } else {
    stop("model must be either a 'varpro' object or an 'rfsrc' grow object.")
  }
  ## ------------------------------------------------------------
  ## Build membership list for prediction cases
  ## ------------------------------------------------------------
  memb_list <- NULL
  n_total   <- NULL
  if (!is.null(newdata)) {
    if (!all(xnames %in% colnames(newdata))) {
      missing_cols <- setdiff(xnames, colnames(newdata))
      stop("newdata is missing required predictors: ", paste(missing_cols, collapse = ", "))
    }
    newx <- newdata[, xnames, drop = FALSE]
    n_total <- nrow(newx)
    pr <- predict.rfsrc(
      rf,
      newx,
      membership = TRUE,
      perf.type  = "none",
      ...
    )
    memb <- pr$membership
    if (is.null(memb)) stop("predict.rfsrc did not return a membership matrix (membership=TRUE).")
    ## map membership rows back to newx rows (handles potential row-dropping)
    if (nrow(memb) != n_total) {
      rn_m <- rownames(memb)
      rn_x <- rownames(newx)
      if (!is.null(rn_m) && !is.null(rn_x)) {
        row_map <- match(rn_m, rn_x)
        if (anyNA(row_map)) stop("Could not align membership rows back to newdata rows.")
      } else {
        stop("predict.rfsrc returned fewer rows than newdata and rownames are unavailable.")
      }
    } else {
      row_map <- seq_len(n_total)
    }
    tree.id <- sort(unique(rule.tree))
    if (max(tree.id) > ncol(memb)) stop("Rule tree indices exceed membership matrix trees.")
    memb_used <- memb[, tree.id, drop = FALSE]
    tree.pos  <- match(rule.tree, tree.id)
    memb_list <- vector("list", R)
    for (tt in seq_along(tree.id)) {
      rules_tt <- which(tree.pos == tt)
      if (!length(rules_tt)) next
      nodes_tt <- memb_used[, tt]
      idx_by_node <- split(row_map, nodes_tt)
      for (r in rules_tt) {
        b <- as.character(rule.branch[r])
        idx <- idx_by_node[[b]]
        memb_list[[r]] <- if (is.null(idx)) integer(0) else as.integer(idx)
      }
    }
  } else {
    ## newdata = NULL: try to reproduce training OOB membership
    if (!is.null(path0$oobMembership)) {
      memb_list <- path0$oobMembership
      if (.is_list_out(object)) {
        n_total <- nrow(object[[1]])
      } else {
        n_total <- nrow(object)
      }
    } else {
      ## reconstruct OOB from restore mode using inbag == 0
      pr <- predict.rfsrc(
        rf,
        membership = TRUE,
        perf.type  = "none",
        ...
      )
      memb  <- pr$membership
      inbag <- pr$inbag
      if (is.null(memb))  stop("predict.rfsrc did not return membership (membership=TRUE).")
      if (is.null(inbag)) stop("predict.rfsrc did not return inbag in restore mode; cannot rebuild OOB.")
      n_total <- nrow(memb)
      tree.id <- sort(unique(rule.tree))
      if (max(tree.id) > ncol(memb)) stop("Rule tree indices exceed membership matrix trees.")
      memb_used  <- memb[,  tree.id, drop = FALSE]
      inbag_used <- inbag[, tree.id, drop = FALSE]
      tree.pos   <- match(rule.tree, tree.id)
      memb_list <- vector("list", R)
      for (tt in seq_along(tree.id)) {
        rules_tt <- which(tree.pos == tt)
        if (!length(rules_tt)) next
        oob_rows <- which(inbag_used[, tt] == 0)
        if (!length(oob_rows)) next
        nodes_oob <- memb_used[oob_rows, tt]
        idx_by_node <- split(oob_rows, nodes_oob)
        for (r in rules_tt) {
          b <- as.character(rule.branch[r])
          idx <- idx_by_node[[b]]
          memb_list[[r]] <- if (is.null(idx)) integer(0) else as.integer(idx)
        }
      }
    }
  }
  ## ------------------------------------------------------------
  ## Aggregate to case x variable gradients using workhorse
  ## ------------------------------------------------------------
  .predict_one <- function(rule_imp, path_spec = NULL) {
    if (is.null(rule_imp)) stop("Missing rule.imp in ivarpro.path (required for prediction).")
    if (length(rule_imp) != R) stop("Length of rule.imp does not match number of rules.")
    csO <- list(
      results       = data.frame(variable = rule.var, imp = as.numeric(rule_imp)),
      xvar.names    = xnames,
      oobMembership = memb_list,
      n             = n_total
    )
    out_j <- csimp.varpro.workhorse(csO, noise.na = noise.na)
    ## Attach path info (preserve ladder gradients if present)
    if (is.null(path_spec)) {
      path_pred <- path0
    } else {
      path_pred <- path_spec
    }
    ## store membership only if requested
    path_pred$oobMembership <- if (isTRUE(path.store.membership)) memb_list else NULL
    attr(out_j, "ivarpro.path") <- path_pred
    out_j
  }
  ## ------------------------------------------------------------
  ## Return object in same shape as input
  ## ------------------------------------------------------------
  if (.is_list_out(object)) {
    ## Detect compact style input: common path on list + per-response rule.imp in element paths
    input_common <- attr(object, "ivarpro.path")
    compact_in <- .has_rule_meta(input_common)
    out <- vector("list", length(object))
    names(out) <- names(object)
    if (compact_in) {
      ## Predict each response using element-specific rule.imp
      for (j in seq_along(object)) {
        rule_imp_j <- .get_rule_imp(object[[j]])
        path_spec  <- attr(object[[j]], "ivarpro.path")
        out[[j]] <- .predict_one(rule_imp_j, path_spec = path_spec)
      }
      ## Attach common path to list-level for compact output
      path_common_pred <- input_common
      path_common_pred$oobMembership <- if (isTRUE(path.store.membership)) memb_list else NULL
      attr(out, "ivarpro.path") <- path_common_pred
    } else {
      ## Legacy input: each element carries full path, including rule.imp
      for (j in seq_along(object)) {
        path_j <- attr(object[[j]], "ivarpro.path")
        if (is.null(path_j)) stop("One element of the ivarpro list is missing 'ivarpro.path'.")
        out[[j]] <- .predict_one(path_j$rule.imp, path_spec = path_j)
      }
    }
    ## carry data (prefer newdata, else reuse stored training data)
    if (!is.null(newdata) && isTRUE(save.data)) {
      attr(out, "data") <- data.frame(newx, check.names = FALSE)
    } else if (isTRUE(save.data) && !is.null(attr(object, "data"))) {
      attr(out, "data") <- attr(object, "data")
    }
    attr(out, "model") <- model
    class(out) <- unique(c("ivarpro", class(out)))
    out
  } else {
    ## Single-output object
    out <- .predict_one(path0$rule.imp, path_spec = path0)
    if (!is.null(newdata) && isTRUE(save.data)) {
      attr(out, "data") <- data.frame(newx, check.names = FALSE)
    } else if (isTRUE(save.data) && !is.null(attr(object, "data"))) {
      attr(out, "data") <- attr(object, "data")
    }
    attr(out, "model") <- model
    class(out) <- unique(c("ivarpro", class(out)))
    out
  }
}
