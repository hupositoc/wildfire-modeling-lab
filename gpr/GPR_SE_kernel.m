function [zhatxstar sigma2xstar] = GPR_SE_kernel(xstar,X,Y,sigma2f,sigma2x)

X = reshape(X',[],1);    

X = reshape(X,2,[])';



sigma2w = 0.3;



n = size(X,1);



KxstarX = GP_K(xstar,X,sigma2f,sigma2x);

KXX = GP_K(X, X, sigma2f, sigma2x);

Kxstarxstar = GP_K(xstar,xstar,sigma2f,sigma2x);

Lambda = (KXX+sigma2w*eye(n))^(-1);

zhatxstar = KxstarX*Lambda*Y;

sigma2xstar = Kxstarxstar - KxstarX * Lambda * KxstarX';