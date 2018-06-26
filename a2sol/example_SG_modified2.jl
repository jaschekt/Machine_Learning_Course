# Load X and y variable
using JLD
data = load("quantum.jld")
(X,y) = (data["X"],data["y"])

# Add bias variable, initialize w, set regularization and optimization parameters
(n,d) = size(X)
lambda = 1

# Initialize
maxPasses = 1
progTol = 1e-4
verbose = true
w = zeros(d,1)
wbar = w
lambda_i = lambda/n # Regularization for individual example in expectation

# Start running stochastic gradient
w_old = copy(w);
m=maxPasses*n

for k in 1:m

    # Choose example to update 'i'
    i = rand(1:n)

    # Compute gradient for example 'i'
    r_i = -y[i]/(1+exp(y[i]*dot(w,X[i,:])))
    g_i = r_i*X[i,:] + (lambda_i)*w

    # Choose the step-size
    alpha = 0.001663

    # Take thes stochastic gradient step
    w -= alpha*g_i

    # Proportionally weighted average
    if k>=n/2
        wbar += 2*w/n
    end


    # Check for lack of progress after each "pass"
    if mod(k,n) == 0
        yXw = y.*(X*w)
        f = sum(log.(1 + exp.(-y.*(X*w)))) + (lambda/2)norm(w)^2
        fbar = sum(log.(1 + exp.(-y.*(X*wbar)))) + (lambda/2)norm(wbar)^2
        delta = norm(w-w_old,Inf);
        if verbose
            @printf("Step size = %f, function = %.4e, change = %.4f, average = %.4e\n",alpha,f,delta,fbar);
        end
        if delta < progTol
            @printf("Parameters changed by less than progTol on pass\n");
            break;
        end
        w_old = copy(w);
    end
end
