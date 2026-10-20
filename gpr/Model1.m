% 울진 지역 위성 데이터 영역 정의
inputpoints = [
    37.0960, 129.2840; % upper left
    37.0240, 129.2840; % lower left
    37.0960, 129.3560; % upper right
    37.0240, 129.3560; % lower right
];

% 그리드 정의
numRows = 100;
numCols = 100;
timeSteps = 50;
maxEpisodes = 100;

latitudes = linspace(inputpoints(1,1), inputpoints(2,1), numRows);
longitudes = linspace(inputpoints(1,2), inputpoints(3,2), numCols);

[LatGrid, LonGrid] = meshgrid(latitudes, longitudes);

% NDVI 데이터 불러오기
filename = 'ndvi_data.csv'; % CSV 파일 이름
try
    ndviData = readtable(filename);
    disp('NDVI 데이터 파일이 성공적으로 불러와졌습니다.');
catch ME
    disp('데이터 파일을 불러오는 중 오류가 발생했습니다.');
    disp(ME.message);
    return;
end

% NDVI 데이터를 그리드 좌표에 맞게 매핑
NDVI = zeros(numRows, numCols);
for i = 1:height(ndviData)
    row = ndviData.row_index(i) + 1;
    col = ndviData.col_index(i) + 1;
    NDVI(row, col) = ndviData.mean(i);
end

% NDVI 정규화 (0~1 범위로 변환)
NDVI = NDVI / max(NDVI(:));

% Threshold 설정
threshold = 0.5; % NDVI가 이 값 이하인 경우 도로로 간주
roadMask = NDVI <= threshold;
NDVI(roadMask) = 0;

% 발화 지점 설정
fireStartLat = 37.03;
fireStartLon = 129.309;

[~, startRow] = min(abs(latitudes - fireStartLat));
[~, startCol] = min(abs(longitudes - fireStartLon));

% 학습 파라미터
alpha = 0.1;
gamma = 0.9;
epsilon = 0.1;
actions = 1:8;
numActions = length(actions);

% Q-table 초기화
Q = zeros(numRows, numCols, numActions);

% 초기 풍향 및 풍속 데이터
initialWindDirection = 90 + randn(numRows, numCols) * 5; % ±5도 노이즈
initialWindSpeed = abs(randn(numRows, numCols) * 3 + 5); % 평균 5m/s

% Time Step 별 풍향과 풍속 생성
windDirection = zeros(numRows, numCols, timeSteps);
windSpeed = zeros(numRows, numCols, timeSteps);

for t = 1:timeSteps
    if t == 1
        windDirection(:, :, t) = initialWindDirection;
        windSpeed(:, :, t) = initialWindSpeed;
    else
        % 이전 값 기반 노이즈 추가
        windDirection(:, :, t) = mod(windDirection(:, :, t-1) + randn(numRows, numCols), 360);
        windSpeed(:, :, t) = max(0, windSpeed(:, :, t-1) + randn(numRows, numCols)); % 음수 방지
    end
end

% 방향 정의 (8방향)
directions = [
    -1,  0, 270; % 상
     1,  0, 90;  % 하
     0, -1, 180; % 좌
     0,  1, 0;   % 우
    -1, -1, 225; % 상좌
    -1,  1, 315; % 상우
     1, -1, 135; % 하좌
     1,  1, 45   % 하우
];

% 보상 함수 조정
ndviRewardWeight = 0.6;
windRewardWeight = 0.3;
windSpeedPenaltyWeight = 0.1;

% 실제 데이터 경로 생성
actual_fire_data = zeros(numRows, numCols, timeSteps);
actual_fire_data(startRow, startCol, 1) = 1;

for t = 2:timeSteps
    actual_fire_data(:, :, t) = actual_fire_data(:, :, t-1);
    for i = 2:numRows-1
        for j = 2:numCols-1
            if actual_fire_data(i, j, t-1) == 1
                for action = 1:numActions
                    ni = i + directions(action, 1);
                    nj = j + directions(action, 2);
                    if ni >= 1 && ni <= numRows && nj >= 1 && nj <= numCols
                        windEffect = exp(-windSpeed(ni, nj, t) / 10);
                        if rand < (NDVI(ni, nj) * windEffect)
                            actual_fire_data(ni, nj, t) = 1;
                        end
                    end
                end
            end
        end
    end
end

% 학습 과정
jaccardIndex = zeros(timeSteps, 1);

for episode = 1:maxEpisodes
    fireSpread = zeros(numRows, numCols);
    fireSpread(startRow, startCol) = 1;

    for t = 1:timeSteps
        newFireSpread = fireSpread;
        anySpread = false;

        for i = 2:numRows-1
            for j = 2:numCols-1
                if fireSpread(i, j) == 1
                    for action = 1:numActions
                        ni = i + directions(action, 1);
                        nj = j + directions(action, 2);

                        if ni >= 1 && ni <= numRows && nj >= 1 && nj <= numCols
                            if NDVI(ni, nj) == 0
                                continue;
                            end

                            windEffect = exp(-windSpeed(ni, nj, t) / 10);
                            if rand < (NDVI(ni, nj) * windEffect)
                                newFireSpread(ni, nj) = 1;
                                anySpread = true;

                                currentWind = windDirection(ni, nj, t);
                                actionDirection = directions(action, 3);
                                angleDifference = abs(currentWind - actionDirection);
                                angleDifference = min(angleDifference, 360 - angleDifference);
                                windReward = cosd(angleDifference);

                                reward = ndviRewardWeight * NDVI(ni, nj) + ...
                                         windRewardWeight * windReward - ...
                                         windSpeedPenaltyWeight * windSpeed(ni, nj, t);

                                maxNextQ = max(Q(ni, nj, :));
                                Q(i, j, action) = Q(i, j, action) + ...
                                    alpha * (reward + gamma * maxNextQ - Q(i, j, action));
                            end
                        end
                    end
                end
            end
        end

        fireSpread = newFireSpread;

        % 시각화
        figure(1);
        subplot(1, 2, 1);
        imagesc(longitudes, latitudes, actual_fire_data(:, :, t));
        title(['Actual Fire Spread (Episode ', num2str(episode), ') - Time Step: ', num2str(t)]);
        colormap(hot);
        colorbar;
        xticks(linspace(min(longitudes), max(longitudes), 5)); % X축: 경도
        yticks(linspace(min(latitudes), max(latitudes), 5));   % Y축: 위도
        xlabel('Longitude');
        ylabel('Latitude');

        subplot(1, 2, 2);
        imagesc(longitudes, latitudes, fireSpread);
        title(['Predicted Fire Spread (Episode ', num2str(episode), ') - Time Step: ', num2str(t)]);
        colormap(hot);
        colorbar;
        xticks(linspace(min(longitudes), max(longitudes), 5)); % X축: 경도
        yticks(linspace(min(latitudes), max(latitudes), 5));   % Y축: 위도
        xlabel('Longitude');
        ylabel('Latitude');

        pause(0.1);

        if ~anySpread
            disp(['Episode ', num2str(episode), ' ended at Time Step ', num2str(t)]);
            break;
        end
    end
end

% 최종 Q-table 시각화
figure(2);
imagesc(Q(:, :, 1));
title('Q-value for Action: Up');
colorbar;

% 가중치 최적화 결과 출력
disp('Time Step별 Jaccard Index:');
disp(jaccardIndex);
