include("misc.jl")
include("findMin.jl")

function tabular(X,y)
	(n,d) = size(X)

	# Compute the frequencies in the training data
	# (The below is *not* efficient in time or space)
	D =  Dict()
	for i in 1:n
		key = [y[i];X[i,:]]
		if haskey(D,key)
			D[key] += 1
		else
			D[key] = 1
		end
	end

	# Sample function
	function sampleFunc(xtilde)
		key0 = [0;xtilde]
		key1 = [1;xtilde]
		p0 = 0
		if haskey(D,key0)
			p0 = D[key0]
		end
		p1 = 0
		if haskey(D,key1)
			p1 = D[key1]
		end
		
		if p0+p1 == 0
			# Probability is undefined, go random
			return rand() < .5
		else
			return rand() < p1/(p0+p1)
		end
	end
	# Return model
	return SampleModel(sampleFunc)
end

function parents(i,j)
    leftboundary  = max(j-2,1)
    upperboundary = max(i-2,1)
    np = (i-upperboundary+1)*(j-leftboundary+1) - 1
    ind = [upperboundary:i,leftboundary:j]
    return ind,np
end

function logReg(X,y)
	(n,d) = size(X)

	# Add bias and convert to sparse for speed
	X = sparse([ones(n,1) X])
	Xt = X'

	# The loss function assumes yi in [-1,1] so convert to this
	y[y .< .5] = -1

	# Initial guess and hyper-parameter
	w = zeros(d+1,1)
	lambda = 1

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,Xt,y,lambda)

	# Fit parameters
	w = findMin(funObj,w,derivativeCheck=false,verbose=false,maxIter=10)

	# Sample function (returns [0,1] values)
	function sampleFunc(xtilde)
		return rand() < 1./(1+exp(-dot([1;xtilde],w)))
	end
	# Return model
	return SampleModel(sampleFunc)
end

function logisticObj(w,X,Xt,y,lambda)
	Z = 1+exp.(-y.*(X*w))
	sigmoid = 1./Z
	f = sum(log.(Z)) + (lambda/2)*dot(w,w)
	g = -(Xt*(y.*(1-sigmoid))) + lambda*w
	return (f,g)
end
