data = load('SPECTF.train.txt');
X = data(:, 2: end) ; y = data(:, 1);

model = svmtrain(y,X);

data_test = load('SPECTF.test.txt');
X_test = data_test(:, 2: end) ; Y_test = data_test(:, 1);

m = size(Y_test,1);

% Group = svmclassify(model,X_test);
[predict_label, accuracy, dec_values] = svmpredict(Y_test , X_test,  model);

fprintf('摁下回车继续');
pause;
% 特征选择
n = size(X,2);
m = size(y,1);
weight_init = zeros(1,n);
for i=1:n
    temp_model = svmtrain( y,X(:,i));
    [predict_label, accuracy, dec_values] = svmpredict(y , X(:,i),  temp_model);
    weight_init(i) = accuracy(1);
end

% 按权重排序[]
weight = weight_init;
order = (1:n);
for i=1:n
    max = -1;
    index = i;
    for j=i:n
        if(weight(j)>max)
            max = weight(j);
            index = j;
        end
    end
    temp = order(i);
    order(i) = order(index);
    order(index) = temp;
    temp = weight(i);
    weight(i) = weight(index);
    weight(index) = temp;
end

fprintf('所有特征按权重由大到小排序：');
for i=1:n
    fprintf('%.0f  ',order(i));
end
fprintf('\n摁下回车进行前向算法\n');
pause;
% 前向算法

sort_X = []; 
sort_X_test = [];
nums = [];
nums_test = [];
% 将X按权重排序重排
for i=1:n
    sort_X = [sort_X   X(:,order(i))];
    temp_model = svmtrain( y,sort_X );
    sort_X_test = [sort_X_test   X_test(:,order(i))];
    % temp_Group = svmclassify(temp_model,sort_X_test);
    [predict_label, accuracy, dec_values] = svmpredict(Y_test , sort_X_test,  temp_model);

    nums_test(i) = accuracy(1);
end
figure;
hold on;
plot(nums_test/100);
xlabel('选取的特征的数量')
ylabel('预测的准确率')
hold off;

fprintf('摁下回车进行后向算法\n');
pause;

sort_X = X; 
sort_X_test = X_test;
% nums = [];
nums_test_b = [];
% 将X按权重排序重排
order = flipud(order);
for i=1:n-1
    sort_X = [sort_X(:,1:order(i)-1)   sort_X(:,order(i)+1:end)];
    temp_model = svmtrain( y,sort_X);
    
    sort_X_test = [sort_X_test(:,1:order(i)-1)   sort_X_test(:,order(i)+1:end)];
    %temp_Group = svmclassify(temp_model,sort_X_test);
    [predict_label, accuracy, dec_values] = svmpredict(Y_test , sort_X_test,  temp_model);
    nums_test_b(i) = accuracy(1);
    
    for k=1:size(order,2)
        if(order(k)>order(i))
            order(k) = order(k)-1;
        end
    end
    
end
figure;
hold on;
 plot([43:-1:1],nums_test_b/100);
xlabel('选取的特征的数量')
ylabel('预测的准确率')
hold off;