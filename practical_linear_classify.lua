require 'torch'

data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}
dataTest = torch.Tensor{
{1,6, 4},
{1,10, 5},
{1,14, 8}
}

X = torch.cat(torch.ones((#data)[1],1),torch.zeros((#data)[1],2),2)
Y = torch.cat(torch.ones((#data)[1]-1),torch.ones(1))
for i = 1,(#data)[1] do
    for j = 2,(#data)[2] do
        X[i][j] = data[i][j]
    end
    Y[i] = data[i][1]
end

theta = torch.cat(torch.ones((#X)[2]-1),torch.ones(1))
-- create linear classify function
function linear_classify(a, b)
    Y_prediction=a*b
    return Y_prediction
end

--create standard deviation function
function standard_deviation(y_real, y_theory)
    error = y_real - y_theory
    error_transpose = torch.cat(torch.ones(1,(#error)[1]-1),torch.ones(1,1))
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
---------------------------------------------------------------------------------------
--define delta Kronecker function
function delta(param1,param2)
    if (param1 - param2)^2 <= 0.25 then
        return 1
    else 
        return 0 
    end
end

Y_cal = torch.cat(torch.ones((#data)[1]-1),torch.ones(1))
--create error function of MIT
function error_MIT(theta_para)
    E = 0
    for i=1,(#Y)[1] do
        E = E + (1 - delta(Y[i],linear_classify(X,theta_para)[i]))
    end
    E = E/((#Y)[1])
    return E
end

--loop calculate theta
function improve_theta(theta_improve)
    while error_MIT(theta_improve) >= 0.001 do
        theta_improve = theta_improve + torch.inverse(X)*Y
        print('theta = ' .. theta_improve[1])
    end
    return theta_improve
end