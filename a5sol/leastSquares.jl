include("misc.jl")

function leastSquares(X,y)

	# Find regression weights minimizing squared error
	w = (X'*X)\(X'*y)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresBias(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)

	# Make linear prediction function
	predict(Xhat) = [ones(size(Xhat,1),1) Xhat]*w

	# Return model
	return GenericModel(predict)
end

function leastSquaresBasis(x,y,p)
	Z = polyBasis(x,p)

	w = (Z'*Z)\(Z'*y)

	predict(xhat) = polyBasis(xhat,p)*w

	return GenericModel(predict)
end

function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end

function weightedLeastSquares(X,y,v)
	V = diagm(v)
	w = (X'*V*X)\(X'*V*y)
	predict(Xhat) = Xhat*w
	return GenericModel(predict)
end