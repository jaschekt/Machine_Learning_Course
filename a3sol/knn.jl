include("misc.jl") # Includes mode function and GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
	k = min(k,n)

  # Let's just pre-compute all the squared distances
  # (increases memory to O(nt), which isn't necessary)
  # (or should change this to use "distancesSquared" in misc.jl")
  D = X.^2*ones(d,t) + ones(n,d)*(Xhat').^2 - 2X*Xhat'

  yhat = zeros(t)
  for i in 1:t
    # Sort the distances to the other points
    nearest = sortperm(D[:,i])

    # Use mode of the labels among the neighbours
    yhat[i] = mode(y[nearest[1:k]])
  end

  return yhat
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end
