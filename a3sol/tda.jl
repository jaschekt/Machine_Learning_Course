include("misc.jl")
include("findMin.jl")

using SpecialFunctions

function tda_predict(Xhat,X,y)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = maximum(y)
  yhat = zeros(t)

  sigma = zeros(d,d,k)
  mu = zeros(d,k)
  dof = 3*ones(1,k)
  nc = zeros(1,k)
  logZ = zeros(1,k)

  for i in 1:k
      yc = find(y.==i)
      nc[i]=length(yc)
      xc = X[yc,:]
     (mu[:,i], dof[i], sigma[:,:,i],logZ[i]) = studentT(xc)
  end

  prob = zeros(t,k)
  for cls in 1:k
      pdf_cls = PDF(Xhat,mu[:,cls],dof[cls],sigma[:,:,cls],logZ[cls])
      prob[:,cls] = nc[cls]/n * pdf_cls
  end

    for w in 1:t
        yhat[w]= indmax(prob[w,:])
    end

  return yhat
end

function tda(X,y)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = tda_predict(Xhat,X,y)
  return GenericModel(predict)
end




function studentT(X)

    (n,d) = size(X)

    # Initialize parameters
    mu = zeros(d)
    Sigma = eye(d)
    dof = 3*ones(1,1)

    # Optimize by block coordinate optimization
    mu_old = ones(d)
    funObj_mu(mu) = NLL(X,mu,Sigma,dof,1)
    funObj_Sigma(Sigma) = NLL(X,mu,Sigma,dof,2)
    funObj_dof(dof) = NLL(X,mu,Sigma,dof,3)
    while norm(mu-mu_old,Inf) > .1
    #for i in 1:5
        mu_old = mu

        # Update mean
        mu = findMin(funObj_mu,mu,verbose=false)

        # Update covariance
        Sigma[:] = findMin(funObj_Sigma,Sigma[:],verbose=false)

        # Update degrees of freedom
        dof = findMin(funObj_dof,dof,verbose=false)
    end

    # Compute square root of log-determinant
    SigmaInv = Sigma^-1
    A = chol(Sigma)'
    logStd = sum(log.(diag(A)))
    dof = dof[1] # Take scalar out of containiner
    logZ = lgamma((dof+d)/2) - (d/2)*log(pi) - logStd - lgamma(dof/2) - (d/2)*log(dof)
    return (mu, dof, SigmaInv,logZ)
end


function PDF(Xhat,mu,dof,SigmaInv,logZ)
    (t,d) = size(Xhat)
    PDFs = zeros(t)

    for i in 1:t
        xCentered = Xhat[i,:] - mu
        tmp = 1 + (1/dof)*dot(xCentered,SigmaInv*xCentered)
        nll = ((d+dof)/2)*log(tmp) - logZ
        PDFs[i] = exp(-nll)
    end
    return PDFs

end

function NLL(X,mu,Sigma,dof,deriv)
    (n,d) = size(X)

    # Initialize nll and gradient
    nll = 0
    if deriv == 1
        g = zeros(d,1)
    elseif deriv == 2
        g = zeros(d,d)
    else
        g = zeros(1,1)
    end

    # Possibly re-shape and symmeetrize sigma
    if size(Sigma,1) != d
        Sigma = reshape(Sigma,d,d)
    end
    Sigma = (Sigma+Sigma')/2

    dof = dof[1]

    if dof < 0
        return (Inf,g[:])
    end

    # Compute a square root of Sigma, and check that it's positive-definite
    A = zeros(d,d)
    try
        A = chol(Sigma)'
    catch
        # Case where Sigma is not positive-definite
        return (Inf,g[:])
    end

    # Take into account log factor
    SigmaInv = Sigma^-1
    for i in 1:n
        xCentered = X[i,:] - mu
        tmp = 1 + (1/dof)*dot(xCentered,SigmaInv*xCentered)
        try
        nll += ((d+dof)/2)*log(tmp)
    catch
        @show d
        @show dof
        @show tmp
        assert(1==0)
    end
        if deriv == 1
            g -= ((d+dof)/(dof*tmp))*SigmaInv*xCentered
        elseif deriv == 2
            g -= ((d+dof)/(2*dof*tmp))*SigmaInv*xCentered*xCentered'*SigmaInv
        else
            g -= (d/(2*tmp*dof^2))*dot(xCentered,SigmaInv*xCentered)
            g -= (dof/(2*tmp*dof^2))*dot(xCentered,SigmaInv*xCentered)
            g += (1/2)*log(tmp)
        end

    end

    # Now take into account normalizing constant
    logStd = sum(log.(diag(A)))
    logZ = lgamma((dof+d)/2) - (d/2)*log(pi) - logStd - lgamma(dof/2) - (d/2)*log(dof)
    nll -= n*logZ
    if deriv == 2
        g += (n/2)*SigmaInv
        g = (g+g')/2
        g = g[:]
    elseif deriv == 3
        g = g - (n/2)*polygamma(0,(dof+d)/2) + (n/2)*polygamma(0,dof/2) + n*(d/(2*dof))
    end

    return (nll,g)
end
