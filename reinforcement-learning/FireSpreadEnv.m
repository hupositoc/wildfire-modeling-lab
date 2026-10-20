classdef FireSpreadEnv < rl.env.MATLABEnvironment
    properties
        GridSize
        NumActions
        FireLocation
        FuelMap
    end
    
    methods
        function this = FireSpreadEnv(grid_size, num_actions)
            obsInfo = rlFiniteSetSpec(1:grid_size^2);
            actInfo = rlFiniteSetSpec(1:num_actions);
            this@rl.env.MATLABEnvironment(obsInfo, actInfo);
            this.GridSize = grid_size;
            this.NumActions = num_actions;
            this.FireLocation = [randi(grid_size), randi(grid_size)];
            this.FuelMap = ones(grid_size);
        end
        
        function [nextState, reward, isDone, loggedSignals] = step(this, action)
            
            move = [0 0; -1 0; 1 0; 0 -1; 0 1; -1 -1; -1 1; 1 -1; 1 1];
            reward = -10; % 기본 패널티
            loggedSignals = [];
            newFireLocation = this.FireLocation + move(action, :);
            if all(newFireLocation >= 1) && all(newFireLocation <= this.GridSize)
                this.FireLocation = newFireLocation;
                fuelAmount = this.FuelMap(this.FireLocation(1), this.FireLocation(2));
                reward = fuelAmount * 10;
                this.FuelMap(this.FireLocation(1), this.FireLocation(2)) = 0; % 연료 소진
            end
            isDone = all(this.FuelMap(:) == 0);
            nextState = sub2ind([this.GridSize, this.GridSize], this.FireLocation(1), this.FireLocation(2));
            this.plotEnvironment()
        end
        
        function initialState = reset(this)
            this.FireLocation = [randi(this.GridSize), randi(this.GridSize)];
            this.FuelMap = ones(this.GridSize);
            initialState = sub2ind([this.GridSize, this.GridSize], this.FireLocation(1), this.FireLocation(2));
        end
        function plotEnvironment(this)
            f = figure(1);
            clf(f);
            hold on;
            imagesc(this.FuelMap); % 연료 맵을 시각화합니다.
            colormap('hot'); % 색상 지정
            scatter(this.FireLocation(2), this.FireLocation(1), 100, 'b', 'filled'); % 현재 화재 위치를 표시
            axis equal;
            title('Fire Spread Environment');
            drawnow; % 화면 갱신
end
    end
end
