function p = predict(theta, X)

    m = size(X, 1); 

    p = zeros(m, 1);


    decision = X*theta;
    for i=1:m,
        if sigmoid(decision(i))>=0.5,
            p(i) = 1;
        else
            p(i) = 0;
        end    
    end

end
