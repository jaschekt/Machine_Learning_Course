# Load X and y variable
using JLD
data = load("groupData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit multi-class logistic regression classifer
include("softmaxClassifierGL1.jl")
#get dimensions
(n,d)=size(X)
k = maximum(y)
lambda = 10
#setup the group for the groupL1regularization
gro = ones(d,k)
for j=1:k
    #set up the groups rowwise
    #ALL different groups
    #gro[:,j]=linspace(1+d*(j-1),d*j,d)
    #ROWWISE same groups
    gro[:,j]=linspace(1,d,d)
    #COLUMNWISE same groups
    #gro[:,j]=j*ones(1,100)
end
model = softmaxClassiferGL1(X,y,lambda,gro)

# Compute training and validation error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)

# Count number of parameters in model and number of features used
nModelParams = sum(model.w .!= 0)
nFeaturesUsed = sum(sum(abs.(model.w),2) .!= 0)
@show(trainError)
@show(validError)
@show(nModelParams)
@show(nFeaturesUsed)

# Show the image as a matrix
using PyPlot
imshow(model.w);
