% 실제 산불 데이터 생성
actual_fire_data = zeros(numRows, numCols, timeSteps);

% 발화 지점
actual_fire_data(startRow, startCol, 1) = 1;

% 풍향 데이터 고정 생성 (Time step별 변화 포함)
windBaseDirection = initialWindDirection; % 초기 풍향 설정
windNoise = randn(1, timeSteps) * 5; % ±5도 범위의 랜덤 노이즈 추가
generatedWindDirection = windBaseDirection + cumsum(windNoise);
generatedWindDirection = mod(generatedWindDirection, 360); % 0~360도 사이로 제한
fixedWindDirection = generatedWindDirection; % 고정된 값으로 설정

% 시뮬레이션
for t = 2:timeSteps
    % 이전 시간의 산불 경로 가져오기
    prevSpread = actual_fire_data(:, :, t-1);
    newSpread = prevSpread;

    % 풍향 데이터 설정 (고정된 데이터 활용)
    currentFixedWindDirection = fixedWindDirection(t);

    for i = 2:numRows-1
        for j = 2:numCols-1
            if prevSpread(i, j) == 1
                for action = 1:numActions
                    ni = i + directions(action, 1);
                    nj = j + directions(action, 2);

                    if ni >= 1 && ni <= numRows && nj >= 1 && nj <= numCols
                        if NDVI(ni, nj) == 0
                            continue;
                        end

                        % NDVI 기반 확산 확률
                        if rand < NDVI(ni, nj)
                            % 풍향 보정
                            actionDirection = directions(action, 3);
                            angleDifference = abs(currentFixedWindDirection - actionDirection);
                            angleDifference = min(angleDifference, 360 - angleDifference);

                            % 풍향 영향을 포함한 확산 결정
                            windInfluence = cosd(angleDifference);
                            if windInfluence > 0.5 % 일정 기준 이상의 풍향 조건일 때만 확산
                                newSpread(ni, nj) = 1;
                            end
                        end
                    end
                end
            end
        end
    end
    actual_fire_data(:, :, t) = newSpread;
end

% 실제 데이터 저장
save('actual_fire_data.mat', 'actual_fire_data', 'fixedWindDirection');

% 시각화
for t = 1:timeSteps
    figure(1);
    imagesc(actual_fire_data(:, :, t));
    title(['Actual Fire Spread (Fixed Wind) - Time Step: ', num2str(t)]);
    colormap(hot);
    colorbar;
    pause(0.1);
end
