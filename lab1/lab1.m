data = load('SPECTF.train.txt');
X = data(:, 2: end) ; y = data(:, 1);

[m, n] = size(X);

X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);

[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

theta = initial_theta;
times = 500000;
alpha = 0.00002;
costs = zeros(1,times);
for i=1:times
    [J, grad] = costFunction(theta, X, y);
    costs(1,i) = J;
    theta = theta - alpha * grad;
end

[J, grad] = costFunction(theta, X, y);

fprintf('Cost at theta found by fminunc: %f\n', J);
fprintf('grad at theta found by fminunc: %f\n', grad);

fprintf('theta: \n');
fprintf(' %f \n', theta);


data_pred = load('SPECTF.test.txt');
X_pred = data_pred(:, 2: end) ; y_pred = data_pred(:, 1);
[m_pred, n_pred] = size(X_pred);

X_pred = [ones(m_pred, 1) X_pred];

p_pred = predict(theta, X_pred);

fprintf('Train Accuracy: %f\n', mean(double(p_pred == y_pred)) * 100);

p = predict(theta, X);
fprintf('原始数据测试结果: %f\n', mean(double(p == y)) * 100);


