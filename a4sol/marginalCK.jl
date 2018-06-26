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
