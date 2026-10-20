classdef MultiAgentFireEnv < rl.env.MATLABEnvironment
    properties
        GridSize
        Agents
        FuelMap
    end

    methods
        function this = MultiAgentFireEnv(grid_size, num_agents)
            % 각 에이전트의 관측치와 행동 정보 설정
            obsInfo = rlNumericSpec([1 grid_size^2]);
            obsInfo.Name = 'Agent Observations';
            actInfo = rlFiniteSetSpec(1:9);  % 8 방향 + 정지
            actInfo.Name = 'Agent Actions';
            
            % 부모 클래스 생성자 호출
            this@rl.env.MATLABEnvironment(repmat(obsInfo, num_agents, 1), repmat(actInfo, num_agents, 1));
            
            this.GridSize = grid_size;
            this.FuelMap = ones(grid_size);
            this.Agents = arrayfun(@(x) [randi(grid_size), randi(grid_size)], 1:num_agents, 'UniformOutput', false);
        end
        
        function [nextState, reward, isDone, loggedSignals] = step(this, actions)
            reward = zeros(numel(this.Agents), 1);
            isDone = false(numel(this.Agents), 1);
            nextState = cell(numel(this.Agents), 1);
            
            for idx = 1:numel(this.Agents)
                action = actions(idx);
                % 에이전트 위치 업데이트 로직 (간략화)
                this.Agents{idx} = updateAgentPosition(this.Agents{idx}, action, this.GridSize);
                % 관측치 업데이트
                nextState{idx} = encodePositionToObservation(this.Agents{idx}, this.GridSize);
            end
            
            % 환경 시각화
            this.plotEnvironment();
        end

        function initialState = reset(this)
            this.Agents = arrayfun(@(x) [randi(this.GridSize), randi(this.GridSize)], 1:numel(this.Agents), 'UniformOutput', false);
            initialState = cellfun(@(pos) encodePositionToObservation(pos, this.GridSize), this.Agents, 'UniformOutput', false);
        end

        function plotEnvironment(this)
            figure(1);
            hold off;
            imagesc(this.FuelMap);
            colormap('hot');
            hold on;
            for i = 1:length(this.Agents)
                agentPos = this.Agents{i};
                plot(agentPos(2), agentPos(1), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
            end
            title('Multi-Agent Fire Spread Environment');
            axis equal;
            grid on;
            drawnow;
        end
    end
end

function observation = encodePositionToObservation(position, gridSize)
    index = sub2ind([gridSize, gridSize], position(1), position(2));
    observation = zeros(1, gridSize * gridSize);
    observation(index) = 1;
end

function updatedPosition = updateAgentPosition(position, action, gridSize)
    % 여기에 에이전트의 위치를 업데이트하는 로직 구현
    % 예시는 간단한 상, 하, 좌, 우 이동
    moves = [0, 1; 0, -1; 1, 0; -1, 0; 1, 1; 1, -1; -1, 1; -1, -1; 0, 0];  % 가능한 이동
    newPosition = position + moves(action, :);
    % 경계 조건 처리
    newPosition = max(1, min(newPosition, gridSize));
    updatedPosition = newPosition;
end
