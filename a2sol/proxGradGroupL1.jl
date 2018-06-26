include("misc.jl")

function proxGradGroupL1(funObj,w,gro,lambda;maxIter=100,epsilon=1e-2)
	# Evalluate the intial objective and gradient
	(f,g) = funObj(w)
	# Initial step size and sufficient decrease parameter
	(gamma,alpha) = [1e-4,1]
	for i in 1:maxIter
		# Gradient step on smoooth part
		wNew = w - alpha*g
		# Proximal step
		wNew = proxStep(wNew ,alpha*lambda,gro)
		(fNew,gNew) = funObj(wNew)
		# Decrease the step-size if we increased the function
		gtd = dot(g,wNew-w)
		while fNew + lambda*groupL1Norm(wNew,gro) > f + lambda*groupL1Norm(w,gro) - gamma*alpha*gtd
			#@printf("Backtracking\n")
			alpha /= 2
			# Try out the smaller step-size
			wNew = w - alpha*g
			wNew = proxStep(wNew ,alpha*lambda,gro)
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
		(w,f,g)=[wNew,fNew,gNew]
		# Print out some diagnostics
		optCond = groupL1Norm(w-proxStep(w-g ,alpha*lambda,gro),gro)
		@printf("%6d %15.5e %15.5e %15.5e\n",i,alpha,f+lambda*groupL1Norm(w,gro),optCond)
		# We want to stop if the gradient is really small
		if optCond < epsilon
			@printf("Problem solved up to optimality tolerance\n")
			return w
		end
	end
	@printf("Reached maximum number of iterations\n")
	return w
end

function  proxStep(w,threshold,gro)
	maxgro = maximum(gro)
	for j in 1:maxgro
		ind = find(x->x==j,gro)
		groupnorm = norm(w[ind])
		if groupnorm !=0
			w[ind] = w[ind]/groupnorm * max(0,groupnorm-threshold);
		end
	end
	return w
end

function groupL1Norm(w,gro)
	maxgro = maximum(gro)
	GL1 = 0
	for j in 1:maxgro
		ind = find(x->x==j,gro)
		GL1 += norm(w[ind])
	end
	return GL1
end
