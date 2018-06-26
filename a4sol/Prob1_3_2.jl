using JLD
include("problem1_functions.jl")
include("sample_backwards.jl")
data = load("markovData.jld")
(p0,pT_long,pT_short1,pT_short2) = (data["p0"],data["pT_long"],data["pT_short1"],data["pT_short2"])
xd=1
n=1000
X = sample_backwards(p0,pT_long,n,xd)
M=computeMarginals(X,p0)
@show(M)
