# Load X and y variable
using JLD
data = load("quantum.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
lambda = 1

# Initialize
maxPasses = 20
progTol = 1e-4
verbose = true
w = zeros(d,1)
v = zeros(n,d)
g = zeros(d,1)
lambda_i = lambda/n # Regularization for individual example in expectation

# Start running stochastic gradient
w_old = copy(w);
    
norms=zeros(n,1)
# Find lipschitz
for i in 1:n
    norms = norm(X[i,:])^2
end
L=.25*max(norms)+lambda

for k in 1:n*maxPasses

    # Choose example to update 'i'
    i = rand(1:n)

    # Compute gradient for example 'i' 
    r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
    g_i = r_i*X[i,:] + (lambda_i)*w

    # Store these gradients
    g = g-v[i,:]+g_i
    v[i,:] = g_i
    #v[i,:] = g_i-v[i,:]
    #g = sum(v,1)
    #g=g[:]

    # Choose the step-size
    alpha = 1/L
        
    # Take thes stochastic gradient step
    w -= alpha*g/(maxPasses*n)
   
   # Check for lack of progress after each "pass"
    if mod(k,n) == 0
        yXw = y.*(X*w)
        f = sum(log.(1 + exp.(-yXw))) + (lambda/2)*norm(w)^2
        change = norm(w-w_old,Inf);
        if verbose
            @printf("pass =%d, alpha = %f, function = %.4e, change = %.4f\n",k/n,alpha,f,change);
        end
        if change < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        w_old = copy(w);
    end
end