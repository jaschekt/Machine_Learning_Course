function leastSquaresEmpericalBayes(X,y)
    (x1,x2) = size(X)
    pvec = 0:10
    range = -7:7
    lambdavec = 10.0.^range
    sigmavec = 10.0.^range
    k = length(pvec)
    n = length(lambdavec)
    m = length(sigmavec)
    NLMmatrix = zeros(n,m)
    NLM = 1E10
    bestlambda = 0
    bestsigma = 0
    bestp = 0
    for dim in 1:k
        p=pvec[dim]
        count=0
        for i in 1:n
            lambda = lambdavec[i]
            for j in 1:m
                sigma = sigmavec[j] 
                Phi = polyBasis(X,p)
                C = eye(x1)*1/sigma^2+Phi*Phi'*1/lambda
                (NLMmatrix[i,j],infstage) = neglogmarginal(y,C,p)
                count = count+infstage
            end
        end
       (optimali,optimalj) = ind2sub(size(NLMmatrix), indmin(NLMmatrix))
        @printf("For degree p = %d, the minimum negative log-marginal prob is %f\n",p,NLMmatrix[optimali,optimalj])
        @printf("There were %d ill-posed C matrices out of a %d by %d parameter grid.\n",count,length(range),length(range))
        if NLMmatrix[optimali,optimalj]<NLM 
            bestp = p
            bestlambda = lambdavec[optimali]
            bestsigma = sigmavec[optimalj]
            NLM = NLMmatrix[optimali,optimalj]
        end
        println("")
    end
    return (bestp,bestlambda,bestsigma,NLM)
end


function neglogmarginal(y,C,p)
    n = size(C)[1]
    if isposdef(C)
        U =chol(C)
        logdetU=0
        count =0
        # Compute the log determinant stabally
        for i in 1:n
            logdetU +=2*log(U[i,i])
        end
        # Compute the y'inv(C)y stabally
        b=U'\y
        x=U\b
        quad = y'*x
        # Put it all together
        NLM = (1/2)*logdetU+(1/2)*quad+(p/2)*log(2*pi)
    else
        NLM =Inf
        count = 1
    end
    NLM = NLM[1]
    return (NLM,count)
end
        
function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end