
require 'torch'
require 'optim'
require 'nn'


----------------------------------------------------------------------
-- 1. Create the training data
--  {watiting time, when}
data = torch.Tensor{
   {4,  10.2},
   {5, 10.4},
   {6, 10.5},
   {4, 10.7},
   {1, 11.9},
   {3, 12.0},
   {6, 12.1},
   {9, 12.9},
   {8, 13.1},
   {8, 13.6}
}

local lamda = 30
local delta = 0.1

X = torch.Tensor((#data)[1], 1):fill(1)
Y = torch.Tensor((#data)[1], 1):fill(1)
for i = 1, (#X)[1] do
    Y[i][1] = data[i][1]
    X[i][1] = data[i][1]
end

local function kernel (x, y)
    return torch.exp(-(1/lamda)*torch.sum(torch.pow(x-y,2)))
end

Phi = torch.Tensor((#data)[1], (#data)[1])
for i = 1, (#data)[1] do
    for j = 1, (#data)[1] do
        Phi[i][j] = kernel(X[{{i}}],X[{{j}}]) 
    end
end

local regularizer = torch.mul(torch.eye((#data)[1]), torch.pow(delta,2))

theta = torch.inverse((Phi:t()*Phi) + regularizer) * Phi:t() * Y

xTest = torch.Tensor{
    {11},
    {10.5},
    {12.5}
}

PhiTest = torch.Tensor((#data)[1], (#xTest)[1])
for i = 1, (#data)[1] do
    for j = 1, (#xTest)[1] do
        PhiTest[i][j] = kernel(X[{{i}}],xTest[{{j}}]) 
    end
end

yPre = PhiTest:t()*theta