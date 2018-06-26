using JLD
include("problem1_functions.jl")
include("forwardBackward.jl")
data = load("markovData.jld")
(p0,pT_long,pT_short1,pT_short2) = (data["p0"],data["pT_long"],data["pT_short1"],data["pT_short2"])
xd=1
Z = forwardBackward(p0,pT_long,xd)
@show(Z)
