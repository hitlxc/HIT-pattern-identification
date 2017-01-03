X = zeros(200,10304);%����

Y = zeros(200,1);%����

for i=1:40
    for j=1:5
        temp = imread(['s',num2str(i),'/',num2str(j),'.pgm']);
        X( (j+(i-1)*5) , : ) = temp(:)';
        Y( (j+(i-1)*5) ) = i;
    end
end

model = svmtrain(Y,X); 

% ����
X_test = zeros(200,10304);%ȡ40��ÿ���е��к�5������200������������
Y_test = zeros(200,1);%����
for i=1:40
    for j=1:5
        temp = imread(['s',num2str(i),'/',num2str(j+5),'.pgm']);
        X_test( (j+(i-1)*5) , : ) = temp(:)';
        Y_test( (j+(i-1)*5) ) = i;
    end
end

[predict_label, accuracy, dec_values] = svmpredict(Y_test , X_test,  model);

fprintf('\n���»س����н�ά\n');
pause;

%PCA��ά
[pc,score,latent,tsquare] = princomp(X);

account = cumsum(latent)./sum(latent);  %��ά��Ŀռ����ܱ�ʾԭ�ռ�ĳ̶�

dimension = size(account,1);

for i = 1 : size(account,1)
    if(account(i)>=0.95 && account(i-1)<0.95)
        dimension = i;
        break;
    end
end

fprintf('��PCAѹ��Ϊ %.0f ά���ܹ���95%%�ĳ̶Ȼ�ԭԭ����\n',dimension);

tranMatrix = pc(:,1:dimension); %���ɷֱ任����

% �������ɷֱ任����tranMatrix���н�ά
X_PCA = X * tranMatrix;
X_test_PCA = X_test * tranMatrix;
model_PCA = svmtrain(Y,X_PCA); 
[predict_label_PCA, accuracy_PCA, dec_values_PCA] = svmpredict(Y_test , X_test_PCA,  model_PCA);

fprintf('%.0f ά��׼ȷ����%f%%\n',dimension,accuracy_PCA(1));

fprintf('\n���»س����и���ά�ȵļ���\n');
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
xlabel('ά��')
ylabel('Ԥ���׼ȷ��')
hold off;