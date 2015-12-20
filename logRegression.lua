require 'torch'

local output = io.open("logRegressionY1.dat", "w+")
local output2 = io.open("logRegressionY0.dat", "w+")
local output1 = io.open("costFunction.dat", "w+")

function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end
dataX = lines_from("ex4Data/ex4x.dat")
dataY = lines_from("ex4Data/ex4y.dat")

Y = torch.Tensor(#dataY,1):fill(1)
X = torch.Tensor(#dataX, 3):fill(1)
theta = torch.Tensor(3, 1):fill(0)
for i = 1, #dataX do
  for j = 2, 3 do 
    X[i][j] = string.split(dataX[i], "   ")[j - 1]
  end
  Y[i] = dataY[i]
end

function linear_classify(x, var_theta)
  Y_prediction=x*var_theta
  return Y_prediction
end

function sigm(linear_fun)
  local temp = torch.Tensor((#linear_fun)[1],1):fill(1)
  for i = 1, (#linear_fun)[1] do
    temp[i] = 1/(1 + torch.exp(-linear_fun[i][1]))
  end
  return temp
end

function log_soft_max(sigm_fun, y)
  local J = 0
  for i = 1, (#sigm_fun)[1] do
    J = J - 1/((#sigm_fun)[1]) * (torch.log(sigm_fun[i][1])*y[i][1] + torch.log(1-sigm_fun[i][1])*(1-y[i][1])) 
  end
  return J
end
result = log_soft_max(sigm(linear_classify(X, theta)), Y)

function gradient_cost_fun(sigm_fun, y, x)
  gradient_J = x:transpose(1,2)*(sigm_fun - y)
  return gradient_J
end



function Hessian_function(sigm_fun, x)
  
  local H = torch.Tensor((#x)[2],(#x)[2]):fill(0)
  
  for i = 1, (#sigm_fun)[1] do
    H = H + (x:transpose(1, 2)*x)*sigm_fun[i][1]*(1 - sigm_fun[i][1])
  end 

  return H
end

function Newton_method(result)
  dem = 0
  while result > 0.1 do
    theta = theta - torch.inverse(Hessian_function(sigm(linear_classify(X, theta)), X))*gradient_cost_fun(sigm(linear_classify(X, theta)), Y, X)
    result = log_soft_max(sigm(linear_classify(X, theta)), Y)
    print(dem,"\t",result)
    dem = dem + 1
    if (dem % 10 == 0) then
      output1:write(dem,"\t",result,"\n")  
    end
  end
  output1:close()
  return result
  
end

