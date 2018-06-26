
# Generate data from a Gaussian with outliers
n = 250
d = 2
nOutliers = 25
mu = randn(d)
Sigma = randn(d,d)
Sigma = (1/2)*(Sigma+Sigma') # Make symmetric
(eigenVals,~) = eig(Sigma)
Sigma += (1-minimum(eigenVals))*eye(d) # Make positive-definite
A = chol(Sigma)' # Get a matrix acting like the "standard deviation": Sigma = A*A'
X = zeros(n,d)
for i in 1:n
    xi = randn(d) # Sample from multivariate standard normal
    X[i,:] = A*xi + mu # Sample from multivariate Gausian (by affine property)
end
X[rand(1:n,nOutliers),:] = abs.(10*rand(nOutliers,d)) # Add some crazy points

include("studentT.jl")
model = studentT(X)

# Plot data and densities (you can ignore the code below)
using PyPlot
plot(X[:,1],X[:,2],".")

increment = 100
(xmin,xmax) = xlim()
xDomain = linspace(xmin,xmax,increment)
(ymin,ymax) = ylim()
yDomain = linspace(ymin,ymax,increment)

xValues = repmat(xDomain,1,length(xDomain))
yValues = repmat(yDomain',length(yDomain),1)

z = model.pdf([xValues[:] yValues[:]])

@assert(length(z) == length(xValues),"Size of model function's output is wrong");

zValues = reshape(z,size(xValues))

contour(xValues,yValues,zValues)
