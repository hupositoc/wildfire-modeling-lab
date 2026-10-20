function [initialState, loggedSignals] = resetFunction()
    initialState = [1; 1];  % 반드시 2x1 열 벡터여야 함
    loggedSignals = [];
end

function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, state)
    state = state(:);  % state를 열 벡터로 보정
    if numel(state) < 2
        error('State dimension is insufficient.');
    end
    gridSize = [5, 5];
    goalState = [5, 5];

    x = state(1);
    y = state(2);
    % ... (나머지 코드는 동일)
end
