%% 1. 상태(State) 및 행동(Action) 정의
% 상태: 2D 좌표 [x; y]
obsInfo = rlNumericSpec([2 1]);
obsInfo.Name = 'AgentPosition';
obsInfo.Description = 'Agent X and Y Position';

% 행동: 1 (Up), 2 (Down), 3 (Left), 4 (Right)
actInfo = rlFiniteSetSpec([1, 2, 3, 4]);
actInfo.Name = 'Action';

%% 2. 상태 전이 함수 (Step Function)
% 이 함수는 행동을 입력받아 다음 상태, 보상, 종료 여부, 로그 데이터를 반환합니다.
function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, state)
    gridSize = [5, 5];    % 격자 크기 5×5
    goalState = [5, 5];   % 목표 상태 (오른쪽 아래 코너)

    % 현재 상태 분해
    x = state(1);
    y = state(2);

    % 선택한 행동에 따른 상태 업데이트
    switch action
        case 1  % Up
            x = max(1, x - 1);
        case 2  % Down
            x = min(gridSize(1), x + 1);
        case 3  % Left
            y = max(1, y - 1);
        case 4  % Right
            y = min(gridSize(2), y + 1);
    end

    nextObs = [x; y];

    % 보상 설정: 목표 상태에 도달하면 +10, 아니면 -1
    if isequal(nextObs', goalState)
        reward = 10;
        isDone = true;
    else
        reward = -1;
        isDone = false;
    end

    loggedSignals = []; % 필요시 추가 정보 기록 가능
end

%% 3. 환경 리셋 함수 (Reset Function)
% 이 함수는 환경을 초기 상태로 재설정하며, 두 개의 출력(초기 상태, 로그 데이터)을 반환해야 합니다.
function [initialState, loggedSignals] = resetFunction()
    initialState = [1; 1]; % 초기 위치 (좌상단)
    loggedSignals = [];
end

%% 4. rlFunctionEnv 객체 생성
env = rlFunctionEnv(obsInfo, actInfo, @stepFunction, @resetFunction);

% 환경 검증 (필요 시)
validateEnvironment(env)