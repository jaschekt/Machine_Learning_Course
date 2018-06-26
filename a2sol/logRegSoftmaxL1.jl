include("misc.jl")
include("findMin.jl")


# Multi-class softmax version (assumes y_i in {1,2,...,k})
function logRegSoftmaxL1(X,y,lambda)
	(n,d) = size(X)
	k = maximum(y)
	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)
	funObj(w) = softmaxL1Obj(w,X,y,k,lambda)
	W[:] = findMinL1(funObj,W[:],lambda,maxIter=500)
	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)
	return LinearModel(predict,W)
end

function softmaxL1Obj(w,X,y,k,lambda)
    #computes objective function and Gradient
	(n,d) = size(X)
	W = reshape(w,d,k)
	XW = X*W
	Z = sum(exp.(XW),2)
	nll = 0
	G = zeros(d,k)
	for i in 1:n
		nll += -XW[i,y[i]] + log(Z[i])
		pVals = exp.(XW[i,:])./Z[i]
		for c in 1:k
			G[:,c] += X[i,:]*(pVals[c] - (y[i] == c))
		end
	end
	return (nll,reshape(G,d*k,1))
end
