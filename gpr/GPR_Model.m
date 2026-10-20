%% 초기화
clear all; close all;

% 주어진 주요 지점 데이터 (위도, 경도, 풍향, 풍속)
data_lat = [37.03, 37.03, 37.03, 37.04, 37.06, 37.06];
data_long = [129.29, 129.31, 129.35, 129.35, 129.33, 129.29];
wind_dir = [270, 290, 310, 300, 260, 250]; % 풍향 (도)
wind_speed = [6.5, 7.2, 5.8, 4.5, 7.0, 6.2]; % 풍속 (m/s)

% 영역 정의
lat_min = 37.03; lat_max = 37.09;
long_min = 129.29; long_max = 129.35;

% 25x25 그리드 생성
n = 25; % 625개의 격자로 나눔
lat_grid = linspace(lat_min, lat_max, n);
long_grid = linspace(long_min, long_max, n);
[lat_mesh, long_mesh] = meshgrid(lat_grid, long_grid);

% 예측할 위치 데이터
predict_points = [reshape(lat_mesh, [], 1), reshape(long_mesh, [], 1)];

%% GPR 수행 함수
function [mean_grid, var_grid] = perform_gpr(predict_points, input_data, target_values, sigma2f, sigma2x, n)
    [mean_values, var_values] = GPR_SE_kernel(predict_points, input_data, target_values, sigma2f, sigma2x);
    mean_grid = reshape(mean_values, n, n); % 평균 값을 2D 그리드로 변환
    var_grid = reshape(diag(var_values), n, n); % 분산 값을 2D 그리드로 변환
end

%% 풍향 데이터에 대한 GPR 수행
sigma2f_dir = 0.1; % 커널 함수의 신호 분산
sigma2x_dir = 0.005; % 커널 함수의 길이 척도
[dir_mean_grid, dir_var_grid] = perform_gpr(predict_points, [data_lat', data_long'], wind_dir', sigma2f_dir, sigma2x_dir, n);

%% 풍속 데이터에 대한 GPR 수행
sigma2f_speed = 0.1; % 커널 함수의 신호 분산
sigma2x_speed = 0.005; % 커널 함수의 길이 척도
[speed_mean_grid, speed_var_grid] = perform_gpr(predict_points, [data_lat', data_long'], wind_speed', sigma2f_speed, sigma2x_speed, n);

%% 시각화 함수
function visualize_with_values(lat_grid, long_grid, data_grid, title_text, colorbar_label)
    figure;
    set(gcf, 'Color', 'w'); % 배경을 흰색으로 설정
    imagesc(long_grid, lat_grid, data_grid);
    colorbar;
    title(title_text);
    xlabel('Longitude');
    ylabel('Latitude');
    set(gca, 'YDir', 'normal'); % Y축 방향을 정상적으로 설정
    % 각 격자 중앙에 숫자 값 표시
    for i = 1:size(data_grid, 1)
        for j = 1:size(data_grid, 2)
            text(long_grid(j), lat_grid(i), sprintf('%.1f', data_grid(i, j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', 'black');
        end
    end
end

%% 풍향 시각화
visualize_with_values(lat_grid, long_grid, dir_mean_grid, 'Predicted Wind Direction (°)', 'Direction (°)');

%% 풍속 시각화
visualize_with_values(lat_grid, long_grid, speed_mean_grid, 'Predicted Wind Speed (m/s)', 'Speed (m/s)');
