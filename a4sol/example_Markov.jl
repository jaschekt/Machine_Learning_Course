# Load X and y variable
using JLD
data = load("markovData.jld")
(p0,pT_long,pT_short1,pT_short2) = (data["p0"],data["pT_long"],data["pT_short1"],data["pT_short2"])

include("exactDecode.jl")
decode_short1 = exactDecode(p0,pT_short1)
decode_short2 = exactDecode(p0,pT_short2)

include("problem1_functions.jl")


X = sampleAncestral(p0,pT_short1,10000)
M_short1 = computeMarginals(X,p0)
X = sampleAncestral(p0,pT_short2,10000)
M_short2 = computeMarginals(X,p0)
X = sampleAncestral(p0,pT_long,10000)
M_long = computeMarginals(X,p0)

display(decode_short1)
display(decode_short2)

print("\n ******* Problem 1.1.1 *******\n")
print("\nMarginals of chain short_1\n")
display(M_short1)
print("gives decoding:", computeDecode(M_short1),"\n")
print("\nMarginals of chain short_2\n")
display(M_short2)
print("gives decoding:", computeDecode(M_short2),"\n")
print("\nMarginals of chain long\n")
display(M_long)
print("gives decoding:", computeDecode(M_long),"\n")

print("\n ******* Problem 1.1.2. *******\n")
M_short1 = marginalCK(p0,pT_short1)
print("\nChapman Kolmogorov with dynamic programming \n")
display(M_short1)
print("gives decoding:", computeDecode(M_short1),"\n")

M_short2 = marginalCK(p0,pT_short2)
print("\nChapman Kolmogorov with dynamic programming \n")
display(M_short2)
print("gives decoding:", computeDecode(M_short2),"\n")

M_long = marginalCK(p0,pT_long)
print("\nChapman Kolmogorov with dynamic programming \n")
display(M_long)
print("gives decoding:", computeDecode(M_long),"\n")

print("\n ******* Problem 1.1.2. *******\n")
print("\nMarginals of chain short_1\n")
display(M_short1)
print("gives decoding:", computeDecode(M_short1),"\n")
print("\nMarginals of chain short_2\n")
display(M_short2)
print("gives decoding:", computeDecode(M_short2),"\n")
print("\nMarginals of chain long\n")
display(M_long)
print("gives decoding:", computeDecode(M_long),"\n")

print("\n ******* Problem 1.1.3. *******\n")
B = viterbiDecode(p0,pT_long)
print("\nViterbi Decoding with dynamic programming \n")
print("gives decoding:", B,"\n")

print("\n ******* Problem 1.2.1. *******\n")
X = sampleAncestral(p0,pT_long,10000)
M_long = computeMarginals(X,p0)
# Normalization accounts for a different number of examples in Y
Y = X[find(x->x==2,X[:,1]),:]
(numsize,densize)=(size(Y,1),size(X,1))
M_start2 = computeMarginals(Y,p0)
normalization = densize/numsize
p2 = (M_start2/M_long[2,1])/normalization
print("\nConditional marginals of chain long using MC\n")
display(p2)

print("\n ******* Problem 1.2.2. *******\n")
# Start using given conditional information
y0 = [0 1]
M_start2 = marginalCK(y0,pT_long)
p2 = M_start2
print("\nConditional marginals of chain long using CK\n")
display(p2)

print("\n ******* Problem 1.2.3. *******\n")
# Start using conditional information
B = viterbiDecode(y0,pT_long)
print("\nViterbi Decoding with x1=2 \n")
print("gives decoding:", B,"\n")

print("\n ******* Problem 1.2.4. *******\n")
# End with conditional information
yT_long = pT_long
yT_long[:,:,end] = [1 0;1 0]
B = viterbiDecode(p0,yT_long)
print("\nViterbi Decoding with xd=1 \n")
print("gives decoding:", B,"\n")

print("\n ******* Problem 1.3.1. *******\n")
X = sampleAncestral(p0,pT_long,10000)
M_long = computeMarginals(X,p0)
# Normalization accounts for a different number of examples in Y
Y = X[find(x->x==1,X[:,end]),:]
(numsize,densize)=(size(Y,1),size(X,1))
M_start2 = computeMarginals(Y,p0)
normalization = densize/numsize
p2 = (M_start2/M_long[1,end])/normalization
print("\nConditional marginals with xd=1 of chain long using MC\n")
display(p2)
print("The number of samples in Y is: ", numsize, "\n")

print("\n ******* Problem 1.3.2. *******\n")
# End with conditional information
y0 = [1 0]
X = sampleBackwards(y0,pT_long,10000)
M_end1 = computeMarginals(X,y0)
print("\nConditional marginals with xd=1 of chain long using backward sampling\n")
display(M_end1)

print("\n ******* Problem 1.3.3. *******\n")
# The code for FB takes in the initial prob, transition probs and the conditional prob
cond = [1 0]
M_long = forwardBackwards(p0,pT_long,cond)
print("\nConditional marginals with xd=1 of chain long using FB algorithm\n")
display(M_long)