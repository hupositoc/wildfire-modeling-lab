clc; clear; close all;

%% 환경 설정
grid_size = 100; % 100x100 격자
max_steps = 100; % 한 에피소드당 최대 스텝 수

% 산불 상태 정의 (0: 비어있음, 1: 나무, 2: 불, 3: 탄 상태)
forest = ones(grid_size, grid_size);  
fire_start = [round(grid_size/2), round(grid_size/2)]; % 중앙에서 발화 시작
forest(fire_start(1), fire_start(2)) = 2; % 중앙에 불(2) 설정

%% 기상 데이터 및 입목도 데이터 로드
data = readtable('generated_weather_data.csv');
weather_matrix = reshape(table2array(data(:,3:6)), grid_size, grid_size, [], 4); % 온도, 풍속, 풍향, 습도

% 입목도 데이터 로드
tree_data = readtable('tree_density_data.csv');
tree_density_matrix = zeros(grid_size, grid_size);
for i = 1:height(tree_data)
    r = tree_data.row_index(i) + 1;
    c = tree_data.col_index(i) + 1;
    tree_density_matrix(r, c) = tree_data.mean(i);
end

% 입목도를 기반으로 나무 상태 초기화
forest(tree_density_matrix < 0.2) = 0; % 입목도가 낮으면 비어있는 공간(0)

%% 셀룰러 오토마타 확산 시뮬레이션
figure;
colormap([1 1 1; 0 1 0; 1 0 0; 0 0 0]); % (흰, 초록, 빨강, 검정)

for t = 1:max_steps
    new_forest = forest; % 다음 상태 저장

    % 현재 시점의 기상 데이터 가져오기
    temp_map = weather_matrix(:,:,t,1); % 온도
    wind_speed = weather_matrix(:,:,t,2); % 풍속
    wind_dir = weather_matrix(:,:,t,3); % 풍향 (각도, 0~360도)
    humidity_map = weather_matrix(:,:,t,4); % 습도

    % 격자 업데이트
    for i = 2:grid_size-1
        for j = 2:grid_size-1
            if forest(i, j) == 2  % 현재 셀이 불타고 있다면
                new_forest(i, j) = 3; % 탄 상태(3)로 변경
                
                % 이웃한 셀 (상하좌우) 탐색
                neighbors = [i-1 j; i+1 j; i j-1; i j+1]; % 4방향 이동
                for k = 1:size(neighbors, 1)
                    ni = neighbors(k, 1);
                    nj = neighbors(k, 2);

                    if forest(ni, nj) == 1  % 나무가 있는 곳만 확산 가능
                        % 기본 확산 확률 설정
                        prob = max(0.3,0.3 + 0.2 * temp_map(ni, nj) - 0.2 * humidity_map(ni, nj));
                        
                        % 바람 영향 고려 (풍향이 일치하는 경우 확산 증가)
                        wind_effect = exp(wind_speed(ni, nj) / 5); % 풍속에 따른 확산 증가
                        angle_to_cell = atan2d(ni - i, nj - j); % 확산 방향 계산
                        angle_diff = abs(wind_dir(ni, nj) - angle_to_cell);
                        if angle_diff < 45  % 바람 방향과 유사하면 확산 증가
                            prob = prob * wind_effect;
                        end

                        % 확률적으로 불이 번짐
                        if rand < prob
                            new_forest(ni, nj) = 2;
                        end
                    end
                end
            end
        end
    end

    % 상태 업데이트
    forest = new_forest;
    
    % 시각화 업데이트
    imagesc(forest);
    title(['Time Step: ', num2str(t)]);
    pause(0.1);
end

disp("Simulation completed!");
