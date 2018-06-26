#create data:
#3 classes, 2 features
#number of examples per class
n1=200
n2=400
n3=800
d=2

#offset and transformation per class
mu1 = zeros(d)
A1 = 0.5*eye(d)
mu2 = [2.5,2.5]
A2 = [0.25 0.0; 0.0 1.0]
mu3 = [-2.5, 1.25]
A3 = 0.7071 * [1 -1; 1 1] * [1.5 0.0; 0.0 0.25]

#true covariance
Sigma1 = A1*A1'
Sigma2 = A2*A2'
Sigma3 = A3*A3'

#allocate arrays for all training data
n=n1+n2+n3
X = zeros(n,d)
y = Array{Int64}(n)

#create training data for all three classes
count = 0
for i in 1:n1
    count = count + 1
    x = randn(d)
    X[count,:] = mu1' + (A1*x)'
    y[count] = 1
end
for i in 1:n2
    count = count + 1
    x = randn(d)
    X[count,:] = mu2' + (A2*x)';
    y[count] = 2
end
for i in 1:n3
    count = count + 1
    x = randn(d)
    X[count,:] = mu3' + (A3*x)';
    y[count] = 3
end


using PyPlot

#first figure, plot points and contours from GDA
figure(1)
plot(X[1:n1,1],X[1:n1,2],".")
plot(X[(n1+1):(n1+n2),1],X[(n1+1):(n1+n2),2],".")
plot(X[(n1+n2+1):n,1],X[(n1+n2+1):n,2],".")

include("gda.jl")
model = GDA(X,y)
yhat = model.predict(X)    #also does plotting of contours
trainError = mean(yhat .!= y)
@printf("Train Error with GDA: %.3f\n",trainError)

#second figure, plot points and contours from TDA
figure(2)
plot(X[1:n1,1],X[1:n1,2],".")
plot(X[(n1+1):(n1+n2),1],X[(n1+1):(n1+n2),2],".")
plot(X[(n1+n2+1):n,1],X[(n1+n2+1):n,2],".")

include("tda.jl")
model = TDA(X,y)
yhat = model.predict(X)    #also does plotting of contours
trainError = mean(yhat .!= y)
@printf("Train Error with TDA: %.3f\n",trainError)
