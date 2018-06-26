include("misc.jl") # Includes mode function and GenericModel typedef

function generativeGaussianSSL_predict(Xhat,X,y,Xbar)
  # set number of iterations of EM
  maxIter = 4

  #get dimensions
  (n,d)   = size(X)
  (n2,d2) = size(Xbar)
  (t,d)   = size(Xhat)
  k       = maximum(y)

  # initialize parameters for multivariate gaussian distribution using GDA
  sigma = zeros(d,d,k)
  mu = zeros(d,k)
  mix = zeros(1,k)
  (mix,mu,sigma) = gda_parameters(X,y)

  #run expectation maximization
  for i in 1:maxIter
      (mix, mu, sigma) = EM(mix,mu,sigma,X,y,Xbar,i)
  end

  #make prediction
  yhat = zeros(t)
  for w in 1:t
      argmax = zeros(1,k)
      for j in 1:k
          detlog = logdet(sigma[:,:,j])
          vec = Xhat[w,:]-mu[:,j]
          argmax[j] = mix[j] - 0.5*detlog -0.5*vec'*(sigma[:,:,j])^(-1)*vec
      end
      yhat[w]= indmax(argmax)
  end
  return yhat
end

function generativeGaussianSSL(X,y,Xbar)
  predict(Xhat) = generativeGaussianSSL_predict(Xhat,X,y,Xbar)
  return GenericModel(predict)
end

function gda_parameters(X,y)
  # same as gda but we just care for the parameters
  (n,d) = size(X)
  k = maximum(y)
  sigma = zeros(d,d,k)
  mu = zeros(d,k)
  nc = zeros(1,k)
  mix = zeros(1,k)
  for i in 1:k
      yc = find(y.==i)
      nc[i]=length(yc)
      mu[:,i] = sum(X[yc,:],1)/nc[i]
      for j in yc
            vec = X[j,:]-mu[:,i]
            sigma[:,:,i] += vec*vec'
      end
      sigma[:,:,i] = sigma[:,:,i]./nc[i]
  end
  mix = nc/n
  return mix,mu,sigma
end

function EM(mix,mu,sigma,X,y,Xbar,iter)
    (n,d) = size(X)
    (t,d) = size(Xbar)
    k = maximum(y)

    # initialize new parameters
    new_sig = zeros(d,d,k)
    new_mu = zeros(d,k)
    new_mix = zeros(1,k)
    r = zeros(t,k)
    nc = zeros(1,k)

    # perform the EM update as introduced in problem 2.1

    # E-step
    @printf("E-step iteration %d *** ",iter)
    print("computing the responsibilities...\n")
    for i in 1:t
        denom = 0
        for c in 1:k
            r[i,c] = mix[c]*gaussianPDF(Xbar[i,:],mu[:,c],sigma[:,:,c])
            denom += r[i,c]
        end
        r[i,:] = r[i,:]/denom
    end

    # M-step
    @printf("M-Step iteration %d *** ",iter)
    print("updating parameters... ")
    for j in 1:k
        yc = find(y.==j)
        nc = length(yc)
        total = nc + sum(r[:,j])
        new_mix[j] = total/(n+t)
        new_mu[:,j] = (sum(X[yc,:],1) + sum(r[:,j].*Xbar,1))./total
        for m in yc
            vec = X[m,:]-mu[:,j]
            new_sig[:,:,j] += vec*vec'
        end
        for m in 1:t
            vec = Xbar[m,:]-mu[:,j]
            new_sig[:,:,j] += r[m,j].*vec*vec'
        end
        new_sig[:,:,j] = new_sig[:,:,j]/total
    end
    print("return parameters...\n")
    return new_mix,new_mu,new_sig
end

function gaussianPDF(xvec,mu,sigma)
    d = length(xvec)
    y = xvec-mu
    p = -d/2*log(2*pi)-0.5*logdet(sigma) -0.5*y'*sigma^(-1)*y
    return exp(p)
end

############################################################################################################################

function generativeGaussianSSLimpute_predict(Xhat,X,y,Xbar)
  # set number of iterations of EM
  maxIter = 2

  #get dimensions
  (n,d) = size(X)
  (n2,d2) = size(Xbar)
  (t,d) = size(Xhat)
  k = maximum(y)

  # initialize parameters for multivariate gaussian distribution using GDA
  sigma = zeros(d,d,k)
  mu = zeros(d,k)
  mix = zeros(1,k)
  (mix,mu,sigma) = gda_parameters(X,y)
  ybar = zeros(n2,1)

  # run expectation maximization with imputation
  for i in 1:maxIter
   # here we use imputation to create more labelled examples
    probs = zeros(1,k)
    for ii in  1:n2
        for j in 1:k
            probs[j] = gaussianPDF(Xbar[ii,:],mu[:,j],sigma[:,:,j])
        end
        ybar[ii] = indmax(probs)
    end
    ybar = trunc.(Int,ybar)
    (mix, mu, sigma) = EM_inpute(mix,mu,sigma,X,y,Xbar,ybar,i)
  end

  #make prediction
  yhat = zeros(t)
  for w in 1:t
      argmax = zeros(1,k)
      for j in 1:k
          detlog = logdet(sigma[:,:,j])
          vec = Xhat[w,:]-mu[:,j]
          argmax[j] = mix[j] - 0.5*detlog -0.5*vec'*(sigma[:,:,j])^(-1)*vec
      end
      yhat[w]= indmax(argmax)
  end
  return yhat
end

function generativeGaussianSSLinpute(X,y,Xbar)
  predict(Xhat) = generativeGaussianSSLimpute_predict(Xhat,X,y,Xbar)
  return GenericModel(predict)
end


function EM_inpute(mix,mu,sigma,X,y,Xbar,ybar,iter)
    new_X = [X;Xbar]
    new_y = [y;ybar]
    (n,d) = size(new_X)
    k = maximum(new_y)

    # initialize new parameters
    new_sig = zeros(d,d,k)
    new_mu = zeros(d,k)
    new_mix = zeros(1,k)

    # perform the EM update as introduced in problem 2.1

    # M-step
    @printf("M-Step iteration %d *** ",iter)
    print("updating parameters... ")
    for j in 1:k
        yc = find(new_y.==j)
        nc = length(yc)
        new_mix[j] = nc/n
        new_mu[:,j] = sum(new_X[yc,:],1)/nc
        for m in yc
            vec = new_X[m,:]-mu[:,j]
            new_sig[:,:,j] += vec*vec'
        end
        new_sig[:,:,j] = new_sig[:,:,j]/nc
    end
    print("return parameters...\n")
    return new_mix,new_mu,new_sig
end
