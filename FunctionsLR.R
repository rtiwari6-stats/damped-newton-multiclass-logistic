# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. 
  ## You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  n = length(y)
  p = ncol(X)
  ntest = length(yt)
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if(!identical(X[,1], rep(1,n))){
    stop("The first column of X should be all 1s.")
  }
  if(!identical(Xt[,1], rep(1, nrow(Xt)))){
    stop("The first column of Xt should be all 1s.")
  }
  # Check for compatibility of dimensions between X and Y
  if(nrow(X) != n){
    stop("The number of rows in X and y should be n.")
  }
  
  # Check for compatibility of dimensions between Xt and Yt
  if(nrow(Xt) != length(yt)){
    stop("The number of rows in Xt and yt should be ntest.")
  }
  
  # Check for compatibility of dimensions between X and Xt
  if(ncol(X) != ncol(Xt)){
    stop("Number of columns in X and Xt must be equal.")
  }
  
  # Check eta is positive
  if(eta <= 0){
    stop("Eta must be positive.")
  }
  
  # Check lambda is non-negative
  if(lambda < 0){
    stop("Lambda must be non-negative.")
  }
  
  #Get training labels
  K = max(y) + 1 #K is not zero indexed
  if( K-1 < max(yt)){
    stop('yt has more labels than y.')
  }
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if(is.null(beta_init)){
    beta_init = matrix(rep(0,p*K),p,K) 
  }
  else{
    if(nrow(beta_init) != p){
      stop('Number of rows in beta_int must be equal to the number of columns in X and Xt.')
    }
    if(ncol(beta_init) != K){
      stop('number of columns of beta_int must be equal to number of rows in Y. ')
    }
  }
  #Initializations
  error_train = rep(0, numIter+1)
  error_test = rep(0, numIter+1)
  objective = rep(0, numIter+1)
  
  #train sigmoid
  sigmoid_train = exp(X %*% beta_init)
  sigmoid_train = sigmoid_train / rowSums(sigmoid_train)
  # test sigmoid
  sigmoid_test = exp(Xt %*% beta_init)
  sigmoid_test = sigmoid_test / rowSums(sigmoid_test)
  #initial train class assignment
  ltrain = apply(sigmoid_train, 1, which.max) - 1
  # initial test class assignment
  ltest = apply(sigmoid_test, 1, which.max) - 1
  error_train[1] = mean(ltrain != y)* 100
  error_test[1] = mean(ltest != yt)  * 100
  
  # Let's compute the indicator
  uniqueY = sort(unique(y))
  ytrain = matrix(0, nrow(X), length(uniqueY))
  for(k in 1:ncol(beta_init)){
    ytrain[uniqueY[k] == y, k] = 1
  }
  
  objective[1] =  - sum(ytrain * log(sigmoid_train)) + (lambda/2) * sum(beta_init * beta_init)
  
  for(k in 1:numIter){
    W = sigmoid_train * (1-sigmoid_train) #our weighted matrix
    # Update beta
    for(j in 1:ncol(beta_init)){
      hessian = solve(crossprod(X, X * W[, j]) + (lambda * diag(rep(1, ncol(X)))))
      beta_init[, j] = beta_init[, j] - eta * hessian %*% ((t(X) %*% (sigmoid_train[, j]-ytrain[, j])) + lambda * beta_init[, j])
    }
    #train sigmoid
    sigmoid_train = exp(X %*% beta_init)
    sigmoid_train = sigmoid_train / rowSums(sigmoid_train)
    # test sigmoid
    sigmoid_test = exp(Xt %*% beta_init)
    sigmoid_test = sigmoid_test / rowSums(sigmoid_test)
    #compute train and test errors
    ltrain = apply(sigmoid_train, 1, which.max) - 1
    ltest = apply(sigmoid_test, 1, which.max) - 1
    error_train[k+1] = mean(ltrain != y)* 100
    error_test[k+1] = mean(ltest != yt)  * 100
    objective[k+1] =  - sum(ytrain * log(sigmoid_train)) + (lambda/2) * sum(beta_init * beta_init)
  }
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta_init, error_train = error_train, error_test = error_test, objective =  objective))
  
}