# Load X and y variable
using JLD
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])
include("problem2_functions.jl")

m = size(X,1)
n = size(X,3)
t = size(Xtest,3)

# Training
# Put all 784 models in dictionary
models = Dict()
print("***** Training phase ***** \n")
for i=1:m
    for j=1:m
        (ind,np)=parents(i,j)
        # Setup supervised learning problem
        ybar = X[i,j,:]
        Xbar = X[ind[1],ind[2],:]
        # Put in matrix form and remove last line as it is not a parent
        Xbar = reshape(Xbar,(np+1,n))'
        Xbar = Xbar[:,1:(end-1)]
        #solve it
        models[(i-1)*m+j] = tabular(Xbar,ybar)
    end
    @ printf "progress: %.1f " i/m*100
    print("% \n")
end

println("Done with training")

# Prediction
using PyPlot
figure(1)
for image in 1:4
    subplot(2,2,image)
    # Grab a random test example
    index = rand(1:t)
    I = Xtest[:,:,index]
    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                (ind,np)=parents(i,j)
                Xtilde = I[ind[1],ind[2]]
                Xtilde = reshape(Xtilde,(np+1,1))'
                Xtilde = Xtilde[1:(end-1)]
                I[i,j] = models[(i-1)*m+j].sample(Xtilde)
            end
        end
    end
    imshow(I)
end
