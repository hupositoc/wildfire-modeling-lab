% 예제: predictedState와 actualState 비교 및 보상 함수 계산

% 625개의 격자를 가진 25x25 그리드 생성
numRows = 25;
numCols = 25;

% 시뮬레이션에서 예측된 산불 확산 상태 (가상 데이터)
predictedState = rand(numRows, numCols) > 0.8; % 예: 랜덤하게 발화 상태 생성

% 실제 산불 확산 상태 (가상 데이터)
actualState = rand(numRows, numCols) > 0.85; % 예: 다른 랜덤 발화 상태 생성

% 입목도 데이터 (환경 보상에 활용, 0~1)
NDVI = rand(numRows, numCols);

% 풍향 데이터 (가상, 각도 값)
windDirection = rand(numRows, numCols) * 360;

% 유사성 계산 (Mean Absolute Error 기반)
differenceMatrix = abs(predictedState - actualState); % 격자별 차이
meanAbsoluteError = sum(differenceMatrix, 'all') / numel(predictedState); % 평균 차이
similarity = 1 - meanAbsoluteError; % 유사성 (1에 가까울수록 유사)

% 환경 보상 계산 (입목도와 풍향에 의존)
envReward = NDVI .* cosd(windDirection); % 입목도와 풍향의 상관성

% 최종 보상 계산 (유사성 기반 보상과 환경 보상의 가중합)
alpha = 0.5; % 유사성 보상의 가중치
beta = 0.5;  % 환경 보상의 가중치
finalRewardMatrix = alpha * similarity + beta * envReward; % 최종 보상 행렬

% 시각화
figure;
subplot(1, 3, 1);
imagesc(predictedState);
title('Predicted State');
colorbar;

subplot(1, 3, 2);
imagesc(actualState);
title('Actual State');
colorbar;

subplot(1, 3, 3);
imagesc(finalRewardMatrix);
title('Final Reward Matrix');
colorbar;
