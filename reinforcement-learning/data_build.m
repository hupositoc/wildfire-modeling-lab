clc; clear; close all;

% 파라미터 설정
grid_size = 10000; % 10000개의 그리드
num_timesteps = 100; % 생성할 타임스텝 수
hour_step = linspace(0, 24, num_timesteps); % 하루(0~24시) 내 시간 분포

% 초기 데이터 저장 변수
Temperature = zeros(grid_size, num_timesteps);
Wind_speed = zeros(grid_size, num_timesteps);
Wind_direction = zeros(grid_size, num_timesteps);
Humidity = zeros(grid_size, num_timesteps);

% 낮과 밤을 고려한 온도 설정
T_day_max = 30; % 낮 최고 온도
T_night_min = 15; % 밤 최저 온도

for i = 1:grid_size
    Temperature(i, :) = T_night_min + (T_day_max - T_night_min) * (sin(pi * (hour_step - 6) / 24) + 1) / 2;
    Temperature(i, :) = Temperature(i, :) + randn(size(Temperature(i, :))); % 약간의 랜덤 노이즈 추가
    
    % 풍속 설정 (낮에 강하고 밤에 약한 패턴)
    Wind_speed_mean = 5 + 3 * sin(pi * (hour_step - 6) / 24);
    Wind_speed(i, :) = Wind_speed_mean + randn(size(Wind_speed_mean));
    Wind_speed(i, Wind_speed(i, :) < 0) = 0; % 풍속이 음수가 되지 않도록 보정
    
    % 풍향 설정 (일정한 방향에서 변동 추가)
    Wind_direction(i, :) = mod(180 + 30 * sin(pi * (hour_step - 6) / 12) + 10 * randn(size(hour_step)), 360);
    
    % 습도 설정 (온도와 반비례하는 패턴)
    Humidity(i, :) = 90 - (Temperature(i, :) - T_night_min) * (75 / (T_day_max - T_night_min));
    Humidity(i, :) = Humidity(i, :) + 5 * randn(size(Humidity(i, :))); % 랜덤 노이즈 추가
    Humidity(i, Humidity(i, :) < 0) = 0; % 습도가 음수가 되지 않도록 보정
    Humidity(i, Humidity(i, :) > 100) = 100; % 습도가 100%를 초과하지 않도록 보정
end

% 데이터 저장
weather_data = [];
for t = 1:num_timesteps
    grid_indices = (1:grid_size)';
    time_col = repmat(hour_step(t), grid_size, 1);
    temp_col = Temperature(:, t);
    wind_speed_col = Wind_speed(:, t);
    wind_dir_col = Wind_direction(:, t);
    humidity_col = Humidity(:, t);
    weather_data = [weather_data; table(grid_indices, time_col, temp_col, wind_speed_col, wind_dir_col, humidity_col, ...
        'VariableNames', {'GridID', 'Hour', 'Temperature', 'WindSpeed', 'WindDirection', 'Humidity'})];
end

% 데이터 시각화 (예제로 5개의 랜덤 그리드만 표시)
random_grids = randperm(grid_size, 5);
figure;
for i = 1:5
    subplot(5,1,i);
    plot(hour_step, Temperature(random_grids(i), :)); grid on;
    ylabel(['Grid ' num2str(random_grids(i)) ' Temp (°C)']);
end
xlabel('Hour');
title('Temperature Variation Across Random Grids');

% 데이터 출력 및 저장
writetable(weather_data, 'generated_weather_data.csv');
disp('Weather data generated for 10000 grids and saved as generated_weather_data.csv');