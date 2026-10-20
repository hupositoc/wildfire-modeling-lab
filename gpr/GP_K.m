function K = GP_K(X1, X2, sigma2f, sigma2x)
    % GP_K: Squared Exponential Kernel
    % Inputs:
    %   X1, X2 - 두 데이터셋의 입력 (n x d 크기의 행렬)
    %   sigma2f - Kernel의 signal variance
    %   sigma2x - Kernel의 length scale
    % Output:
    %   K - 커널 행렬

    % 차원의 차이를 계산
    dist = pdist2(X1, X2).^2;

    % Squared Exponential Kernel 공식
    K = sigma2f * exp(-0.5 * dist / sigma2x);
end
