# Generate 5 lags
library(MASS)
library(mvtnorm)
n <- 5000
set.seed(0819)
#m <- 5
m = 15
eps <- rnorm(n,0,1)
X <- matrix(0, ncol=m, nrow=n)
rho = 0.75
X[,1] <- rnorm(n,0,1)
# m is time
# time dependent
mean_x = seq(1:15)
mean_t = cos(0.5 * mean_x)
for(j in 2:m){
X[,j] <- rnorm(n, mean=rho*mean[j], sd = 0.08) 
}



# baseline covariates (Steingrimsson et al. 2019)
d <- 10
Sig <- matrix(0,d,d)
rho2 = 0.9
for(a in 1:d){
  for(b in 1:d){
	if(a==b){
		Sig[a,b] <- 1
	}else{
 		Sig[a,b] <- rho2^(abs(a-b))
	}
  }
}

V <- mvrnorm(n, mu = rep(0,d), Sig = Sig)
#Z <- cbind(exp(z[,1]), z[,3]^2, as.numeric(z[,5] > 0))
# treatment
Z <- rbinom(n,1,0.5)
# construct data
beta0 <- 0.5
beta_z <- 1
beta1 <- c(1,0.5,-0.3,0.4,1)
gamma1 <- c(0.5,0.7,-0.4)
gamma2 <- c(0.4,0.1,-0.7)
eps <- rf(n, df1 = 2, df= 10, ncp= 2)

y_true_z1 <- beta0 + beta_z + (X %*% beta1)^2 + sin(V[,c(1,3,5)] %*% gamma1) + cos(V[,c(2,4,6)] %*% gamma2) + eps
y_true_z0 <- beta0 + (X %*% beta1)^2 + sin(V[,c(1,3,5)] %*% gamma1) + eps

y <- Z*y_true_z1 + (1-Z)*y_true_z0

dat<- data.frame(y=y, Z=Z, X=X, V=V)
# training data/test data
index <- sample(1:n,size=n/2)
train.dat <- dat[index,]
test.dat <- dat[-index,]

head(train.dat)
write.csv(train.dat, "train.csv", row.names = FALSE)
write.csv(test.dat, "test.csv", row.names = FALSE)

# impute to create potential outcomes
# deep learning
# control shift c 
# object <- deep learning model(y ~ .)
# test_y_obs_z1 <- y[Z==1]
# Z.new <- 1-Z
# pred <- pred(object,newdata = data.frame(1-Z,X,V))
# test_y_miss_z1 <- pred[Z.new==1]
# test_y_obs_z0 <- y[Z==0]
# test_y_miss_z0 <- pred[Z.new==0]
# 
# trt_effect <- test_y_miss_z1 - test_y_miss_z0
# 
# # deep learning
# # subgroup (find complex function that affects treatment effect)
# model(trt_effect ~ X,V (test data)) # if possible, cross-validation





