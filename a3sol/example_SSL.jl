# Load X and y variable
using JLD
data = load("SSL.jld")
(X,y,Xtest,ytest,Xbar) = (data["X"],data["y"],data["Xtest"],data["ytest"],data["Xbar"])

# Fit a KNN classifier
k = 5
include("knn.jl")
model = knn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)

# Fit GDA model
include("gda.jl")
model = gda(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with GDA: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with GDA: %.3f\n",testError)

# Fit generative gaussian SSL
include("generativeGaussianSSL.jl")
model = generativeGaussianSSL(X,y,Xbar)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("\nTrain Error with generative Gaussian SSL: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("\nTest Error with generative Gaussian SSL: %.3f\n",testError)

# Fit generative gaussian SSL with imputation
model = generativeGaussianSSLinpute(X,y,Xbar)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("\nTrain Error with generative Gaussian SSL using imputation: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("\nTest Error with generative Gaussian SSL using imputation: %.3f\n",testError)

