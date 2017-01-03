X = zeros(200,10304);%特征

Y = zeros(200,1);%分类

for i=1:40
    for j=1:5
        temp = imread(['s',num2str(i),'/',num2str(j),'.pgm']);
        X( (j+(i-1)*5) , : ) = temp(:)';
        Y( (j+(i-1)*5) ) = i;
    end
end

model = svmtrain(Y,X); 

% 测试
X_test = zeros(200,10304);%取40组每组中的中后5个，共200个做测试样本
Y_test = zeros(200,1);%分类
for i=1:40
    for j=1:5
        temp = imread(['s',num2str(i),'/',num2str(j+5),'.pgm']);
        X_test( (j+(i-1)*5) , : ) = temp(:)';
        Y_test( (j+(i-1)*5) ) = i;
    end
end

[predict_label, accuracy, dec_values] = svmpredict(Y_test , X_test,  model);

fprintf('\n摁下回车进行降维\n');
pause;

%PCA降维
[pc,score,latent,tsquare] = princomp(X);

account = cumsum(latent)./sum(latent);  %降维后的空间所能表示原空间的程度

dimension = size(account,1);

for i = 1 : size(account,1)
    if(account(i)>=0.95 && account(i-1)<0.95)
        dimension = i;
        break;
    end
end

fprintf('用PCA压缩为 %.0f 维后能够以95%%的程度还原原数据\n',dimension);

tranMatrix = pc(:,1:dimension); %主成分变换矩阵

% 利用主成分变换矩阵tranMatrix进行降维
X_PCA = X * tranMatrix;
X_test_PCA = X_test * tranMatrix;
model_PCA = svmtrain(Y,X_PCA); 
[predict_label_PCA, accuracy_PCA, dec_values_PCA] = svmpredict(Y_test , X_test_PCA,  model_PCA);

fprintf('%.0f 维的准确率是%f%%\n',dimension,accuracy_PCA(1));

fprintf('\n摁下回车进行更高维度的计算\n');
pause;

% loop_dimensions = int16(linspace(1,dimension,20));
loop_dimensions = int16(linspace(1,size(account,1),20));
accuracys = [];
x_index = [];
for i = 1 : size(loop_dimensions,2)
    tranMatrix = pc(:,1:loop_dimensions(i));
    X_PCA = X * tranMatrix;
    X_test_PCA = X_test * tranMatrix;
    model_PCA = svmtrain(Y,X_PCA); 
    [predict_label_PCA, accuracy_PCA, dec_values_PCA] = svmpredict(Y_test , X_test_PCA,  model_PCA);
    x_index = [x_index i];
    accuracys = [accuracys accuracy_PCA(1) ];
end

figure;
hold on;
plot(loop_dimensions,accuracys);
xlabel('维度')
ylabel('预测的准确率')
hold off;