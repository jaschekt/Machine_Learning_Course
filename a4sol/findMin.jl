include("misc.jl")

function findMin(funObj,w;maxIter=100,epsilon=1e-2,derivativeCheck=false,verbose=true)
	# funObj: function that returns (objective,gradient)
	# w: initial guess
	# maxIter: maximum number of iterations
	# epsilon: stop if the gradient gets below this
	# derivativeCheck: whether to check against numerical gradient

	# Evalluate the intial objective and gradient
	(f,g) = funObj(w)

	# Optionally check if gradient matches finite-differencing
	if derivativeCheck
		g2 = numGrad(funObj,w)

		if maximum(abs.(g-g2)) > 1e-4
			@show([g g2])
			@printf("User and numerical derivatives differ\n")
			sleep(1)
		else
			@printf("User and numerical derivatives agree\n")
		end
	end

	# Initial step size and sufficient decrease parameter
	gamma = 1e-4
	alpha = 1
	for i in 1:maxIter

		# Try out the current step-size
		wNew = w - alpha*g
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gg = dot(g,g)
		while fNew > f - gamma*alpha*gg

			if verbose
				@printf("Backtracking\n")
			end

			if isfinitereal(fNew)
				# Fit a degree-2 polynomial to set step-size
				alpha = alpha^2*gg/(2(fNew - f + alpha*gg))
			else
				alpha /= 2
			end
			
			# Try out the smaller step-size
			wNew = w - alpha*g
			(fNew,gNew) = funObj(wNew)
		end

		# Guess the step-size for the next iteration
		alphaOld = alpha
		y = gNew - g
		alpha *= -dot(y,g)/dot(y,y)

		# Sanity check on the step-size
		if (!isfinitereal(alpha)) | (alpha < 1e-10) | (alpha > 1e10)
			alpha = 1
		end

		# Accept the new parameters/function/gradient
		w = wNew
		f = fNew
		g = gNew

		# Print out some diagnostics
		gradNorm = norm(g,Inf)
		if verbose
			@printf("%6d %15.5e %15.5e %15.5e\n",i,alphaOld,f,gradNorm)
		end

		# We want to stop if the gradient is really small
		if gradNorm < epsilon
			if verbose
				@printf("Problem solved up to optimality tolerance\n")
			end
			return w
		end
	end
	if verbose
		@printf("Reached maximum number of iterations\n")
	end
	return w
end



function findMinL1(funObj,w,lambda;maxIter=100,epsilon=1e-2)
	# funObj: function that returns (objective,gradient)
	# w: initial guess
	# lambda: value of L1-regularization parmaeter
	# maxIter: maximum number of iterations
	# epsilon: stop if the gradient gets below this

	# Evalluate the intial objective and gradient
	(f,g) = funObj(w)

	# Initial step size and sufficient decrease parameter
	gamma = 1e-4
	alpha = 1
	for i in 1:maxIter

		# Gradient step on smoooth part
		wNew = w - alpha*g
		# Proximal step on non-smooth part
		wNew = sign.(wNew).*max.(abs.(wNew) - lambda*alpha,0)
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gtd = dot(g,wNew-w)
		while fNew + lambda*norm(wNew,1) > f + lambda*norm(w,1) - gamma*alpha*gtd
			@printf("Backtracking\n")
			alpha /= 2

			# Try out the smaller step-size
			wNew = w - alpha*g
			wNew = sign.(wNew).*max.(abs.(wNew) - lambda*alpha,0)
			(fNew,gNew) = funObj(wNew)
		end

		# Guess the step-size for the next iteration
		y = gNew - g
		alpha *= -dot(y,g)/dot(y,y)

		# Sanity check on the step-size
		if (!isfinitereal(alpha)) | (alpha < 1e-10) | (alpha > 1e10)
			alpha = 1
		end

		# Accept the new parameters/function/gradient
		w = wNew
		f = fNew
		g = gNew

		# Print out some diagnostics
		optCond = norm(w-sign.(w-g).*max.(abs.(w-g)-lambda,0),Inf)
		@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f+lambda*norm(w,1),optCond)

		# We want to stop if the gradient is really small
		if optCond < epsilon
			@printf("Problem solved up to optimality tolerance\n")
			return w
		end
	end
	@printf("Reached maximum number of iterations\n")
	return w
end

function findMinGroupL1(funObj,w,lambda,groups;maxIter=100,epsilon=1e-2)
	# funObj: function that returns (objective,gradient)
	# w: initial guess
	# lambda: value of group L1-regularization parmaeter
	# groups: list of group number for each variable
	# maxIter: maximum number of iterations
	# epsilon: stop if the gradient gets below this

	# Evalluate the intial objective and gradient
	(f,g) = funObj(w)

	# Initial step size and sufficient decrease parameter
	gamma = 1e-4
	alpha = 1
	for i in 1:maxIter

		# Gradient step on smoooth part
		wNew = w - alpha*g
		# Proximal step on non-smooth part
		wNew = groupSoftThreshold(wNew,lambda*alpha,groups)
		(fNew,gNew) = funObj(wNew)

		# Decrease the step-size if we increased the function
		gtd = dot(g,wNew-w)
		while fNew + lambda*groupNorm(wNew,groups) > f + lambda*groupNorm(w,groups) - gamma*alpha*gtd
			@printf("Backtracking\n")
			alpha /= 2

			# Try out the smaller step-size
			wNew = w - alpha*g
			wNew = groupSoftThreshold(wNew,lambda*alpha,groups)
			(fNew,gNew) = funObj(wNew)

			assert(alpha > 1e-10)
		end

		# Guess the step-size for the next iteration
		y = gNew - g
		alpha *= -dot(y,g)/dot(y,y)

		# Sanity check on the step-size
		if (!isfinitereal(alpha)) | (alpha < 1e-10) | (alpha > 1e10)
			alpha = 1
		end

		# Accept the new parameters/function/gradient
		w = wNew
		f = fNew
		g = gNew

		# Print out some diagnostics
		optCond = norm(w-groupSoftThreshold(w-alpha*g,lambda,groups),Inf)
		@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f+lambda*groupNorm(w,groups),optCond)

		# We want to stop if the gradient is really small
		if optCond < epsilon
			@printf("Problem solved up to optimality tolerance\n")
			return w
		end
	end
	@printf("Reached maximum number of iterations\n")
	return w
end

function groupNorm(w,groups)
	d = length(w)
	nGroups = maximum(groups)
	v = zeros(nGroups)
	for j in 1:d
		v[groups[j]] += w[j]^2
	end
	return sum(sqrt.(v))
end

function groupSoftThreshold(w,threshold,groups)
	d = length(w)
	nGroups = maximum(groups)
	v = zeros(nGroups)
	for j in 1:d
		v[groups[j]] += w[j]^2
	end
	v = sqrt.(v)
	wNew = zeros(d,1)
	for g in 1:nGroups
		if v[g] != 0
			wNew[groups.==g] = (w[groups.==g]./v[g])*maximum([0 v[g]-threshold])
		end
	end
	return wNew
end
