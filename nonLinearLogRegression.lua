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
dataX = lines_from("MachineLearning/ex5Data/ex5Logx.dat")
dataY = lines_from("MachineLearning/ex5Data/ex5Logy.dat")

Y = torch.Tensor(#dataY,1):fill(1)
X = torch.Tensor(#dataX, 28):fill(1)
theta = torch.Tensor(28, 1):fill(0)

lamda = 10

for i = 1, #dataX do
  for j = 2, 3 do 
    X[i][j] = string.split(dataX[i], ",")[j - 1]
  end
  Y[i] = dataY[i]
end

for i = 1, #dataX do
  temp_new = 0
  temp_old = 0
  j = 0
  while j <= 6 do
    temp_new = temp_old + j + 1
    for k = temp_old + 1 , temp_new do
      e = k - temp_old -1
      X[i][k] = torch.pow(X[i][2], (j - e)) * torch.pow(X[i][3], e)
    end
    temp_old = temp_new
    j = j + 1
  end
end

function hypothesis_function(x, var_theta)
  Y_prediction=x*var_theta
  return Y_prediction
end

function sigm(hypothesis_func)
  local temp = torch.Tensor((#hypothesis_func)[1],1):fill(1)
  for i = 1, (#hypothesis_func)[1] do
    temp[i] = 1/(1 + torch.exp(-hypothesis_func[i][1]))
  end
  return temp
end

function log_soft_max(sigm_fun, y, var_theta)
  local J = 0
  for i = 1, (#sigm_fun)[1] do
    J = J - 1/((#sigm_fun)[1]) * (torch.log(sigm_fun[i][1])*y[i][1] + torch.log(1-sigm_fun[i][1])*(1-y[i][1]))
  end
  for i =1, (#var_theta)[1] do
    J = J + lamda/(2*((#var_theta)[1]))*var_theta[i][1]
  end
  return J
end
result = log_soft_max(sigm(hypothesis_function(X, theta)), Y, theta)

function gradient_cost_fun(sigm_fun, y, x, var_theta)
  gradient_J = x:transpose(1,2)*(sigm_fun - y) + var_theta*lamda/((#sigm_fun)[1])
  return gradient_J
end



function Hessian_function(sigm_fun, x)
  
  local H = torch.Tensor((#x)[2],(#x)[2]):fill(0)
  
  for i = 1, (#sigm_fun)[1] do
    H = H + (x:transpose(1, 2)*x)*sigm_fun[i][1]*(1 - sigm_fun[i][1])
  end 
  H = H + torch.eye(28)*lamda/((#sigm_fun)[1])

  return H
end

function Newton_method(result)
  dem = 0
  while result > 0.0001 do
    theta = theta - torch.inverse(Hessian_function(sigm(hypothesis_function(X, theta)), X))*gradient_cost_fun(sigm(hypothesis_function(X, theta)), Y, X, theta)
    result = log_soft_max(sigm(hypothesis_function(X, theta)), Y, theta)
    print(dem,"\t",result)
    dem = dem + 1
    if (dem % 10 == 0) then
      output1:write(dem,"\t",result,"\n")  
    end
  end
  output1:close()
  return result
  
end

function decision_boundary(var_theta)
  b = torch.Tensor(1, (#X)[1]):fill(1)
  a = var_theta:transpose(1,2)
  local x = torch.gels(zero, a):t()
  for i = 1, (#X)[1] do
    output:write(X[i][2], "\t", X[i][3], "\t", x[i][2], "\t", x[i][3], "\n")
  end
  output:close()
  return x 
end

