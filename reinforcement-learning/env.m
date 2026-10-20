classdef env < rl.env.MATLABEnvironment
    % 산불 확산 예측을 위한 강화학습 환경

    %% Properties
    properties
        GridSize        % 격자 크기
        State           % 현재 상태 (불의 위치)
        ActionSpace     % 가능한 행동 공간 (8방향 이동 + 정지)
        Reward          % 보상 값 저장
        Vegetation      % 입목도 (0~1 사이의 값)
        WindDirection   % 풍향 정보
        Humidity        % 습도 정보 (0~1 사이 값)
    end

    %% Methods
    methods
        % Constructor
        function this = env(gridSize)
            if nargin < 1 || isempty(gridSize)
                gridSize = [20, 20]; % 기본 크기 설정
            end

            % 행동 공간 정의 (8방향 + 정지)
            actionInfo = rlFiniteSetSpec(1:9);
            actionInfo.Name = "Fire Spread Direction";

            % 🔹 상태 공간 정의 (1D 벡터로 변환)
            numStates = prod(gridSize) * 3;  % 20x20x3 → 1200
            observationInfo = rlNumericSpec([numStates, 1], ...
                'LowerLimit', 0, 'UpperLimit', 1);
            observationInfo.Name = "Wildfire Grid State with Environment";

            % 상위 클래스 생성자 호출
            this@rl.env.MATLABEnvironment(observationInfo, actionInfo);

            % 격자 크기 설정
            this.GridSize = gridSize;
            this.State = zeros(gridSize);
            this.Vegetation = rand(gridSize); % 입목도 (0~1 랜덤값)
            this.WindDirection = randi([1, 9], gridSize); % 바람 방향
            this.Humidity = 0.5 * ones(gridSize); % 습도 (기본값 0.5)

            % 행동 공간 설정 (8방향 + 정지)
            this.ActionSpace = [
                -1, -1; -1,  0; -1,  1;
                 0, -1;  0,  0;  0,  1;
                 1, -1;  1,  0;  1,  1
            ];

            this.Reward = 0; % 보상 초기화
        end

        % Reset function: 초기 상태 설정
        function initialState = reset(this, fireStart)
            this.State = zeros(this.GridSize);
            if nargin < 2 || isempty(fireStart)
                fireStart = [10, 10]; % 중앙에서 시작
            end
            this.State(fireStart(1), fireStart(2)) = 1;

            % 🔹 3D 상태 데이터를 1D 벡터로 변환
            initialState = reshape(cat(3, this.State, this.Vegetation, this.Humidity), [], 1);
        end

        % Step function: 행동 적용하여 환경 업데이트
        function [obs, reward, isDone, info] = step(this, actionIdx)
            [nextState, reward, isDone, info] = stepImpl(this, actionIdx);

            % 🔹 3D 상태 데이터를 1D 벡터로 변환
            obs = reshape(cat(3, this.State, this.Vegetation, this.Humidity), [], 1);
        end
    end
end
