include("misc.jl") # Includes mode function and GenericModel typedef

function gda_predict(Xhat,X,y)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = maximum(y)

  sigma = zeros(d,d,k)
  mu = zeros(d,k)
  nc = zeros(1,k)

  for i in 1:k
      yc = find(y.==i)
      nc[i]=length(yc)
      mu[:,i] = sum(X[yc,:],1)/nc[i]
      for j in yc
          sigma[:,:,i] = (X[j,:]-mu[:,i])*(X[j,:]-mu[:,i])'+sigma[:,:,i]
      end
      sigma[:,:,i] = sigma[:,:,i]/nc[i]
  end

  yhat = zeros(t)
  for w in 1:t
      argmax = zeros(1,k)
      for t in 1:k
          argmax[t] = nc[t]/n  -0.5*( logdet(sigma[:,:,t]) ) -0.5* (  (Xhat[w,:]-mu[:,t])'*  (sigma[:,:,t])^(-1) * (Xhat[w,:]-mu[:,t]))
      end
      yhat[w]= indmax(argmax)
  end

  return yhat
end

function gda(X,y)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = gda_predict(Xhat,X,y)
  return GenericModel(predict)
end
