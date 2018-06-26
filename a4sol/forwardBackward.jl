include("problem1_functions.jl")

function forwardBackward(p0,pT,xd)
    M=marginalCK(p0,pT)
    k = length(p0)
    d = size(pT,3)+1
    V = backward(xd,p0,pT)
    Z = M.*V
    for j in 1:d
        Z[:,j] = Z[:,j]/sum(Z[:,j])
    end
    return Z
end


function backward(xd,p0,pT)
    k = length(p0)
    d = size(pT,3)+1
    V = zeros(k,d)
    V[xd,d]=1
    for i in d-1 :-1 :1
        V[:,i] = pT[:,:,i]*V[:,i+1]
    end
    return V
end
