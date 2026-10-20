function wd = GPR_SE()

close 
clear all

%% 데이터를 불러온다 (울진일대 위도 경도와 온도를 담고 있는 데이터)

load 'wd_data_matrix.mat';

nr_data = size(z, 1); 

data_lat = z(:,1);

data_long = z(:,2);

data_wind = z(:,3); 


%% 데이터를 약간 가공한다.



% 위치 데이터를 정리한다. 

data_x = (data_lat)*1;

data_y = (data_long)*1;

x_min = min(data_x); x_max = max(data_x);

y_min = min(data_y); y_max = max(data_y);

x_interval = x_max - x_min; 

y_interval = y_max - y_min;



% 10%의 마진을 주고 x,y의 lim을 구한다.

x_lim = [x_min - x_interval/10 x_max + x_interval/10];

y_lim = [y_min - y_interval/10 y_max + y_interval/10];



% Gaussian Process를 위해서 temperature의 mean을 0으로 한다.

% Gaussian Process는 일반적으로 zero-mean을 가정한다. 

mean_temp = mean(data_wind);

data_wind = data_wind - mean_temp;


%% GP 로 regresion할 데이터들을 초기화한다.



% 1. GP를 할 위치들이다.

% 우리가 알고 있는 data들을 기반으로 이 새로운 위치에서 온도를 predict한다. 이 때 Gaussian Process Regression을 사용한다. 

% GPR은 우리가 알고 있는 데이터와 새로운 위치가 Gaussian distribution을 갖는다고 가정을 하고 해당 위치에서 온도를 예측. 

res_gp = (min(x_interval, y_interval/20)); % GP 할 때 resolution을 의미한다. (위치 사이의 간격)

x_gp = x_lim(1):res_gp:x_lim(2);

y_gp = y_lim(1):res_gp:y_lim(2);

[x_cord, y_cord] = meshgrid(x_gp, y_gp); % 위치를 2차원 grid로 만든다. 

X = [reshape(x_cord, [], 1), reshape(y_cord, [], 1)];



% 2. 데이터를 설정한다. (모두 column vector이다.)

Xbar = reshape([data_x';data_y'], [], 1);

Ybar = data_wind;



%% Gaussian Process Regression을 수행한다. 

sigma2f = 0.3;         % GP kernel의 gain을 의미한다. 

sigma2x = .005;        % GP kernel의 var를 의미한다. 

[zhatxstar, sigma2xstar] = GPR_SE_kernel(X, Xbar, Ybar, sigma2f, sigma2x);



%% GPR 결과를 plot한다. 

zhatxstar = zhatxstar + mean_temp; % GPR에 원래 mean을 더해준다. (앞에서 zero mean으로 바꾸기 위해서 평균을 뺐었다.)



option.main_title = 'Gaussian Process Regression on temperature in Seoul @ 2012.11.08';

option.mean_title = 'Temperature mean in Seoul';

option.var_title = 'GPR variance in Seoul';

option.xlabel = 'Longitude';

option.ylabel = 'Latitude';

option.data_x = data_x;

option.data_y = data_y;

mean_gp = reshape(zhatxstar, size(y_gp,2), size(x_gp,2));

var_gp = reshape(diag(sigma2xstar),size(y_gp,2),size(x_gp,2));



%figGP = plot_mean_variance(x_gp, y_gp, mean_gp, var_gp, option);




%% 변수 정리 (단위는 위도, 경도, 온도 이다.)

% X: GPR을 수행할 x,y들의 집합 

% Xbar: 측정한 위치 (x, y)

% Ybar: 측정한 값 

% x_gp: x값 만의 벡터 (imagesc에서 사용된다)

% y_gp: y값 만의 벡터 

% mean_gp: 2차원 행렬에 gp로 regression한 mean value들이 들어가있다. 

%           이 값이 이제 reference가 될 것이다. 

% var_gp: 2차원 행렬에 gp로 regression한 variance value들이 들어있다. 

 

%% 