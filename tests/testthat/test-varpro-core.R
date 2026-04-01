test_that("varpro fit, importance, and predict work on bundled data", {
  skip_on_cran()
  data(alzheimers)

  d <- alzheimers[1:180, ]
  set.seed(101)

  o <- varpro(
    Diagnosis ~ .,
    d,
    ntree = 12,
    nvar = 6,
    max.tree = 12,
    max.rules.tree = 10,
    split.weight = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )

  expect_s3_class(o, "varpro")

  imp <- importance(o, plot.it = FALSE)
  expect_type(imp, "list")
  expect_true("unconditional" %in% names(imp))

  p <- predict(o, d[1:12, , drop = FALSE])
  expect_equal(nrow(as.matrix(p)), 12)

  top <- get.topvars(o)
  expect_true(length(top) > 0)

  org <- get.orgvimp(o, pretty = TRUE)
  expect_true(all(c("variable", "z") %in% names(org)))
})

test_that("outpro and outpro.null produce expected distance outputs", {
  skip_on_cran()
  data(alzheimers)

  d <- alzheimers[1:160, ]
  set.seed(102)

  o <- varpro(
    Diagnosis ~ .,
    d,
    ntree = 12,
    nvar = 5,
    max.tree = 12,
    max.rules.tree = 10,
    split.weight = FALSE,
    parallel = FALSE,
    verbose = FALSE
  )

  out <- outpro(
    o,
    newdata = d[1:20, , drop = FALSE],
    neighbor = 5,
    max.rules.tree = 10,
    max.tree = 10
  )

  expect_true(is.list(out))
  expect_true("distance" %in% names(out))
  expect_equal(length(out$distance), 20)

  out0 <- outpro.null(
    o,
    nulldata = d[1:20, , drop = FALSE],
    neighbor = 5,
    max.rules.tree = 10,
    max.tree = 10
  )

  expect_true("cdf" %in% names(out0))
  expect_true("quantile" %in% names(out0))
  expect_equal(length(out0$quantile), length(out0$distance))
})
