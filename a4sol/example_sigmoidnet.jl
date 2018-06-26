# Load X and y variable
using JLD
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])
include("problem2_functions.jl")

m = size(X,1)
n = size(X,3)
t = size(Xtest,3)

# Training
# Create a dictionary for every model
models = Dict()
print("***** Training phase ***** \n")
for i in 1:m
    for j in 1:m
        ybar =X[i,j,:]
        Xbar = X[1:i,1:j,:]
        Xbar = reshape(Xbar,(i*j,n))'
        Xbar = Xbar[:,1:(end-1)]
        models[(i-1)*m+j] = logReg(Xbar,ybar)
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
                Xtilde = I[1:i,1:j]
                Xtilde = reshape(Xtilde,(i*j,1))'
                Xtilde = Xtilde[1:(end-1)]
                I[i,j] = models[(i-1)*m+j].sample(Xtilde)
            end
        end
    end
    imshow(I)
end

figure(2)
for image in 1:4
    subplot(2,2,image)
    # Grab a random test example
    index = rand(1:t)
    I = Xtest[:,:,index]
    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                Xtilde = I[1:i,1:j]
                Xtilde = reshape(Xtilde,(i*j,1))'
                Xtilde = Xtilde[1:(end-1)]
                I[i,j] = models[(i-1)*m+j].sample(Xtilde)
            end
        end
    end
    imshow(I)
end

figure(3)
for image in 1:4
    subplot(2,2,image)
    # Grab a random test example
    index = rand(1:t)
    I = Xtest[:,:,index]
    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                Xtilde = I[1:i,1:j]
                Xtilde = reshape(Xtilde,(i*j,1))'
                Xtilde = Xtilde[1:(end-1)]
                I[i,j] = models[(i-1)*m+j].sample(Xtilde)
            end
        end
    end
    imshow(I)
end

figure(5)
for image in 1:4
    subplot(2,2,image)
    # Grab a random test example
    index = rand(1:t)
    I = Xtest[:,:,index]
    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                Xtilde = I[1:i,1:j]
                Xtilde = reshape(Xtilde,(i*j,1))'
                Xtilde = Xtilde[1:(end-1)]
                I[i,j] = models[(i-1)*m+j].sample(Xtilde)
            end
        end
    end
    imshow(I)
end
