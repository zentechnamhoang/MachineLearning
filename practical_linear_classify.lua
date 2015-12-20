require 'torch'

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

local output = io.open("practical1_linear.dat", "w+")

-- tests the functions above
dataX = lines_from("ex2Data/ex2x.dat")
dataY = lines_from("ex2Data/ex2y.dat")

X = torch.cat(torch.ones(#dataX,1),torch.zeros(#dataX,1),2)
Y = torch.ones(#dataY,1)

theta = torch.ones((#X)[2])

for i = 1, (#Y)[1] do
    X[i][2] = dataX[i]
    Y[i] = dataY[i]
end
-- create linear classify function
function linear_classify(a, b)
    Y_prediction=a*b
    return Y_prediction
end

--create standard deviation function
function standard_deviation(y_real, y_theory)
    error = y_real - y_theory
    error_transpose = torch.ones(1,(#error)[1])
    for i = 1,(#error)[1] do
        error_transpose[1][i]=Y[i]
    end
    
    J = error_transpose*error
    return J
end

--calculate theta by least squares
function least_squares(x1,y1)
    theta_better=torch.inverse((x1:transpose(1,2))*x1)*(x1:transpose(1,2))*y1
    return theta_better
end

least_squares_method = linear_classify(X,least_squares(X,Y))
for i = 1, (#Y)[1] do
    output:write(X[i][2],"\t",Y[i][1],"\t", least_squares_method[i][1], "\n")
end

output:close()