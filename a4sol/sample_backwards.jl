include("problem1_functions.jl")

function sample_backwards(p0,pT,n,xd)
    # compute p(x)
    M=marginalCK(p0,pT)
    # p(y|x)*p(x)/sum p(y|x)*p(x)
    k = length(p0)
    d = size(pT,3)+1
    X = zeros(n,d)
    X[:,d]=xd
    for i in 1:n
        for j in d-1 :-1 :1
            p_numerator = pT[:, Int(X[i,j+1]), j] .* M[:,j]
            p_dominator = sum(p_numerator)
            p_back =  p_numerator /p_dominator
            X[i,j] = sampleDiscrete(p_back)
        end
    end
    return X
end
