function [J, grad] = costFunction(theta, X, y)

    J = 0;
    grad = zeros(size(theta));

    h_theta = 1./(1+exp(-X*theta));
    n = size(X,2);
    for i=1:m
        J = J + (h_theta(i)-y(i))^2; 
        for j=1:n
            grad(j) = grad(j)+(h_theta(i)-y(i))*X(i,j);
        end
    end
    J = J/2;
    grad = grad/m;

end
