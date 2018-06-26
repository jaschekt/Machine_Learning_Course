function sampleAncestral(p0,pT,n)
	k = length(p0)
	d = size(pT,3)+1
	X = zeros(n,d)
	for i in 1:n
		X[i,1] = sampleDiscrete(p0)
		for j in 2:d
			X[i,j] = sampleDiscrete(pT[Int(X[i,j-1]),:,j-1])
		end
	end
	return X
end

function sampleBackwards(p0,pT,n)
    reversed = flipdim(pT,3)
    X = sampleAncestral(p0,reversed,n)
    X = flipdim(X,2)
    return X
end

function forwardBackwards(p0,pT,cond)
    k = length(p0)
    d = size(pT,3)+1
    reversed = flipdim(pT,3)
    M = zeros(k,d)
    V = zeros(k,d)
    # use dynamic programming
    M = chapmanStep(d,p0,pT,M)
    V = chapmanStep(d,cond,reversed,V)
    V = flipdim(V,2)
    Z = M.*V
    for i in 1:d
        Z[:,i] = normalize(Z[:,i],1)
    end
    return Z
end

function sampleDiscrete(p)
	minimum(find(cumsum(p[:]).> rand()))
end

function computeMarginals(X,p0)
	k = length(p0)
	(n,d) = size(X)
 	M = zeros(k,d)
	for i=1:d
		for j = 1:(k-1)
			M[j,i] =  length(find(x->x==j,X[:,i]))/n
		end
		M[k,i] = 1 - sum(M[:,i])
	end
	return M
end

function computeDecode(M)
	(k,d) = size(M)
	D = zeros(Int64,1,d)
	for i in 1:d
		(p,D[i]) = findmax(M[:,i])
	end
	return D
end

function marginalCK(p0,pT)
    k = length(p0)
    d = size(pT,3)+1
    M = zeros(k,d)
    #use dynamic programming
    M = chapmanStep(d,p0,pT,M)
    return M
end

function chapmanStep(j,p0,pT,M)
    if j==1
        M[:,1] = p0
    elseif (M[1,j] == 0)
        previousProb = chapmanStep(j-1,p0,pT,M)[:,j-1]
        M[:,j] = pT[:,:,j-1]'*previousProb
    end
    return M
end

function viterbiDecode(p0,pT)
    k = length(p0)
    d = size(pT,3)+1
    M = zeros(k,d)
    state = zeros(k,d-1)
    B = zeros(1,d)
    M[:,1]=p0
    for i in 2:d
        for u in 1:k # later state
            comparematrix = zeros(1,k)
            for v in 1:k #previous state
                comparematrix[1,v] = pT[v,u,i-1]'*M[v,i-1]
            end
            (M[u,i], state[u,i-1]) = findmax(comparematrix)
        end
    end
    (~,B[1,d]) = findmax(M[:,d])
    #backtrack
    for j in 1:d-1
        previous = Int(B[1,d-j+1])
        B[1,d-j] = state[previous,d-j]
    end
    B = convert(Array{Int64,2},B)
    return B
end
