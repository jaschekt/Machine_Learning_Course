function exactDecode(p0,pT)

    k = length(p0)
    d = size(pT,3)+1

    maxProb = 0
    maxY = zeros(d)

    # Enumerate over possible assignments to statements
    for y in DiscreteStates(d,k)
        # Compute probability of assignment
        p = p0[y[1]]
        for j in 2:d
            #this is a inhomogeneous Markov chain
            #compute probability as product of conditionals
            p *= pT[y[j-1],y[j],j-1]
        end
        if p > maxProb
            maxProb = p
            maxY = y
        end
    end
    return maxY
end


# The code below defines an iterator that goes through all possible states
struct DiscreteStates
    d::Int
    k::Int
end
Base.start(::DiscreteStates) = Int64(0)
Base.done(S::DiscreteStates, state) = state > S.k^S.d-1
function Base.next(S::DiscreteStates, state)
    # NOT AT ALL EFFICIENT!!! d^k possibilities
    s = state
    y = zeros(Int64,S.d)
    for pos in 1:S.d
        y[pos] = 1+div(s,S.k^(S.d-pos))
        s = mod(s,S.k^(S.d-pos))
    end
    return (y,state+1)
end
