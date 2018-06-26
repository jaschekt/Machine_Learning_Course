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
