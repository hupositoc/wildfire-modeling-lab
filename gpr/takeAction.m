function nextState = takeAction(currentState, action, numRows, numCols)
    % 현재 상태에서 행동(action)을 수행한 다음 상태 계산
    nextState = currentState;
    if action == 1 && currentState(1) > 1 % 상
        nextState(1) = currentState(1) - 1;
    elseif action == 2 && currentState(1) < numRows % 하
        nextState(1) = currentState(1) + 1;
    elseif action == 3 && currentState(2) > 1 % 좌
        nextState(2) = currentState(2) - 1;
    elseif action == 4 && currentState(2) < numCols % 우
        nextState(2) = currentState(2) + 1;
    end
end