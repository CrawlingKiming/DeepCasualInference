# Generate 5 lags
library(MASS)
library(mvtnorm)


n <- 5000
set.seed(0819)
#m <- 5
m = 
eps <- rnorm(n,0,1)
X <- matrix(0, ncol=m, nrow=n)
rho = 1.1
X[,1] <- rnorm(n,0,1)
# m is time
# time dependent
mean_x = seq(1:36)
mean_y = seq(1:36)
mean_t = cos(2*(pi /10) * mean_x)*5
for(j in 2:m){
  X[,j] <- rnorm(n, mean=rho*mean_x[j], sd = 0.5) 
}
#X = mvrnorm(n, mu=mean_x, Sig = exp(-cdist(c(mean_x[1:25]), c(mean_y[1:25])) / (2/4)))
#K = mvrnorm(n, mu=mean_x, Sig = exp(-cdist(c(mean_x), c(mean_y)) / (2/4)))

K = K[,36]
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
#beta1 <- c(1,0.5,-0.3,0.4,1,1,0.5,-0.3,0.4,1,1,0.5,-0.3,0.4,1,1,0.5,-0.3,0.4,1,1,0.5,-0.3,0.4,1)
beta <- c(seq(1:25))
gamma1 <- c(0.5,0.7,-0.4)
gamma2 <- c(0.4,0.1,-0.7)
eps <- rnorm(n)

y_true_z1 <- beta0 + beta_z + K + sin(V[,c(1,3,5)] %*% gamma1) + cos(V[,c(2,4,6)] %*% gamma2) + eps
y_true_z0 <- beta0 + K + sin(V[,c(1,3,5)] %*% gamma1) + eps

y <- Z*y_true_z1 + (1-Z)*y_true_z0

dat<- data.frame(y=y, Z=Z, X=X[1:25], V=V)
# training data/test data
index <- sample(1:n,size=n/2)
train.dat <- dat[index,]
test.dat <- dat[-index,]

head(train.dat)
write.csv(train.dat, "C:/Users/ChoDongKyu/PycharmProjects/Causual/data/train_cycle5.csv", row.names = FALSE)
write.csv(test.dat, "C:/Users/ChoDongKyu/PycharmProjects/Causual/data/test_cycle5.csv", row.names = FALSE)