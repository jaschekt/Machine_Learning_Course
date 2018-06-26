# Load X and y variable
using JLD
data = load("MNIST_images.jld")
(X,Xtest) = (data["X"],data["Xtest"])

m = size(X,1)
n = size(X,3)
t = size(Xtest,3)

# change start
# Train an inhomogeneous Markov chain
p_ijk = zeros(m,m,2)
for j in 1:m
    p_ijk[1,j,:] = sum(X[1,j,:] .== 1)/n
end

for i in 2:m
    for j in 1:m
        p_ijk[i,j,1] = sum( (X[i,j,:] .==1) .& (X[i-1,j,:] .==0) )/sum(X[i-1,j,:] .==0)
        p_ijk[i,j,2] = sum( (X[i,j,:] .==1) .& (X[i-1,j,:] .==1) )/sum(X[i-1,j,:] .==1)

    end
end
# change end

# Show  parameters
using PyPlot

# Fill-in some random test images
figure(1)
for image in 1:4
    subplot(2,2,image)

    # Grab a random test example
    ind = rand(1:t)
    I = Xtest[:,:,ind]

    # Fill in the bottom half using the model
    for i in 1:m
        for j in 1:m
            if isnan(I[i,j])
                if I[i-1,j] ==0
                    I[i,j] = rand() < p_ijk[i,j,1]
                else
                    I[i,j] = rand() < p_ijk[i,j,2]
                end
            end
        end
    end
    imshow(I)
end
