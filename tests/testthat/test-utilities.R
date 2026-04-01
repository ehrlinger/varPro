test_that("roughfix imputes missing values and preserves structure", {
  d <- data.frame(
    x = c(1, NA, 3),
    f = factor(c("a", NA, "b"))
  )

  out <- roughfix(d)

  expect_true(all(!is.na(out$x)))
  expect_true(all(!is.na(out$f)))
  expect_equal(names(out), c("x", "f"))
  expect_s3_class(out$f, "factor")
})

test_that("get.mc.cores returns a positive integer", {
  n <- get.mc.cores()

  expect_true(is.numeric(n))
  expect_gte(as.integer(n), 1L)
})

test_that("get.splitweight.custom aligns weights to encoded predictors", {
  d <- data.frame(
    y = c(1, 0, 1, 0),
    x1 = c(1, 2, 3, 4),
    x2 = c(0, 1, 0, 1)
  )

  sw <- get.splitweight.custom(y ~ ., d, namedvec = c(x1 = 0.25, x2 = 0.75))

  expect_true(is.numeric(sw))
  expect_true(length(sw) >= 2)
  expect_true(all(c("x1", "x2") %in% names(sw)))
  expect_equal(as.numeric(sw["x1"]), 0.25)
  expect_equal(as.numeric(sw["x2"]), 0.75)
})
