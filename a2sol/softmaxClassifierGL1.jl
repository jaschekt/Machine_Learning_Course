include("misc.jl")
include("proxGradGroupL1.jl")

# Multi-class softmax version (assumes y_i in {1,2,...,k})
#same exept replace findminL1 by proxGradGroupL1
function softmaxClassiferGL1(X,y,lambda,gro)
	(n,d) = size(X)
	k = maximum(y)
	W = zeros(d,k)
	funObj(w) = softmaxObj(w,X,y,k)
	W[:] = proxGradGroupL1(funObj,W[:],gro,lambda)
	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)
	return LinearModel(predict,W)
end

function softmaxObj(w,X,y,k)
    #computes objective function and Gradient, same as before
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
