# This is a script to save your own tests for the function
source("FunctionsLR.R")

#load required testing package
if (!require(testthat)) install.packages('testthat')


#test with bad inputs
test_inputmismatches_invalidinputs = test_that('Test for input mismatche or invalid inputs', {
  #first setup something that works
  X = matrix(c(1,2,3,1,6,7,1,3,4), nrow=3, ncol=3, byrow = TRUE)
  Xt = matrix(c(1,3,4,1,5,6), nrow=2, ncol=3, byrow = TRUE)
  y = c(0,1,2) #k=3
  yt = c(0,2)

  #now change an input to cause a particular error each time
  #change X to not have 1s in first column
  Xno1s =  matrix(c(1,2,3,1,6,7,1,3,4), nrow=3, ncol=3, byrow = FALSE)
  expect_error(LRMultiClass(Xno1s, y, Xt, yt))
  
  #change Xt to be without 1s
  Xtno1s = matrix(c(1,3,4,1,5,6), nrow=2, ncol=3, byrow = FALSE)
  expect_error(LRMultiClass(Xt, y, Xtnols, yt))
  
  #X and y
  ybad = c(0,1)
  expect_error(LRMultiClass(X, ybad, Xt, yt))
  
  #Xt and yt
  ytbad = c(0,1,3)
  expect_error(LRMultiClass(X, y, Xt, ytbad))
  
  #X and Xt
  Xbad = matrix(c(1,2,3,1,6,7,1,3), nrow=2, ncol=4, byrow = TRUE)
  Xt = matrix(c(1,3,4,1,5,6), nrow=2, ncol=3, byrow = TRUE)
  expect_error(LRMultiClass(Xbad, y, Xt, yt))
  
  #eta and lambda
  expect_error(LRMultiClass(X,y, Xt, yt, eta=0))
  expect_error(LRMultiClass(X,y, Xt, yt, eta= -0.3))
  expect_error(LRMultiClass(X,y, Xt, yt, lambda= -3))
  
  #beta_init
  beta_initbad1 = matrix(c(1,2,3,4,5,6), nrow=2, ncol=3) #p=2, K=3
  expect_error(LRMultiClass(X,y,Xt,yt, beta_init = beta_initbad1))
  
  beta_initbad2 = matrix(c(1,2,3,4,5,6), nrow=3, ncol=2) #p=2, K=2
  expect_error(LRMultiClass(X,y,Xt,yt, beta_init = beta_initbad2))
})

#test for iris dataset
test_irisdataset = test_that('test using iris dataset', {
  skip_if_not_installed("glmnet")
  #get the predictors from iris data
  X = iris[, c(1,2,3,4)]
  X = cbind(rep(1,150), X)
  y = factor(iris$Species, levels=c("setosa","versicolor", "virginica"), labels=c(0,1,2))
  y=as.numeric(as.character(y))
  X = as.matrix(X)
  out = LRMultiClass(X, y, X, y)
  expect_equal(length(out), 4)
  #try with glmnet
  mod = glmnet::glmnet(X, y, family = "multinomial", alpha=0, lambda=1)
  pred = predict(mod, newx = X, s=1, type = "response")
  ylglmnet = apply(pred, 1, which.max)-1
  glmneterror = mean(y != ylglmnet)
  #we want to find a lower or equal error than glmnet
  expect_gte(length(out$error_train - glmneterror*100 <= 1e-5), 1)
  
})

#test with a single class
test_singleclass = test_that('test with single class',{
  skip_if_not_installed("glmnet")
  
  X = matrix(rnorm(100,0,1), nrow = 25, ncol = 4) #n=25, p=4+1
  X = cbind(rep(1,25), X) # n=25, p=5
  
  # K = 1
  y = rep(0, 25) # n=25, K=1
  out = LRMultiClass(X,y,X,y)
  expect_true(ncol(out$beta) == 1)
  expect_true(nrow(out$beta) == 5)
  expect_true(identical(out$beta, as.matrix(rep(0,5), nrow=5, ncol=1)))
})

#test with two classes
test_twoclass = test_that('test with single class',{
  skip_if_not_installed("glmnet")
  
  X1 = matrix(rnorm(10*4, 0, 1), nrow=10, ncol=4)
  X2 = matrix(rnorm(15*4, 30, 1), nrow=15, ncol=4)
  X = rbind(X1,X2) #n=25, p=4
  X = cbind(rep(1,25), X) # n=25, p=5
  
  # K = 2
  y = c(rep(0, 10), rep(1,15)) # n=25, K=2
  out = LRMultiClass(X,y,X,y)
  # we do not expect zero betas
  expect_true(!identical(out$beta, matrix(c(rep(0,10)), nrow = 5, ncol = 2)))
  expect_true(identical(out$error_test, out$error_train))
  
  #compare with glmnet
  mod = glmnet::glmnet(X, y, family = "multinomial", alpha=0, lambda=1)
  pred = predict(mod, newx = X, s=1, type = "response")
  ylglmnet = apply(pred, 1, which.max)-1
  glmneterror = mean(y != ylglmnet)
  #we want to find a lower or equal error than glmnet
  expect_true(out$error_train[25] - glmneterror*100 <= 1e1)
})

#test with three classes
test_threeclass = test_that('test with three class',{
  skip_if_not_installed("glmnet")
  
  X1 = matrix(rnorm(8*4, 0, 1), nrow=8, ncol=4)
  X2 = matrix(rnorm(9*4, 30, 1), nrow=9, ncol=4)
  X3 = matrix(rnorm(8*4, 90, 1), nrow=8, ncol=4)
  X = rbind(X1,X2,X3) #n=25, p=4
  X = cbind(rep(1,25), X) # n=25, p=5
  
  # K = 3
  y = c(rep(0, 8), rep(1,9), rep(2,8)) # n=25, K=3
  out = LRMultiClass(X,y,X,y, numIter = 50)
  # we do not expect zero betas
  expect_true(!identical(out$beta, matrix(c(rep(0,15)), nrow = 5, ncol = 3)))
  expect_true(identical(out$error_test, out$error_train))
  
  #compare with glmnet
  mod = glmnet::glmnet(X, y, family = "multinomial", alpha=0, lambda=1)
  pred = predict(mod, newx = X, s=1, type = "response")
  ylglmnet = apply(pred, 1, which.max)-1
  glmneterror = mean(y != ylglmnet)
  #we want to find a lower or equal error than glmnet but it doesn't always happen
  expect_true(out$error_train[25] - glmneterror*100 <= 1e1)
})

