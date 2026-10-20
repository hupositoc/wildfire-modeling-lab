A=imread('ULJIN.png','png'); %Uljin.png 사진을  배열  형식으로  불러옴  [X,Y,RGB 값] [a,b]=size(A)
figure
%specify the coordinate (lat,long)
inputpoints(1,:)=[37.0960,129.2840];%upperleft
inputpoints(2,:)=[37.0240,129.2840];%lowerleft
inputpoints(3,:)=[37.0960,129.3560];%upperright
inputpoints(4,:)=[37.0240,129.3560];%lowerright
basepoints(1,:)=[1,a];
basepoints(2,:)=[1,1];
basepoints(3,:)=[b,a];
basepoints(4,:)=[b,1];
tform=cp2tform(inputpoints,basepoints,'affine'); 
Xhat=inputpoints(3,1):(inputpoints(2,1)-inputpoints(3,1))/100:inputpoints(2,1); Yhat=inputpoints(2,2):(inputpoints(3,2)-inputpoints(2,2))/10:inputpoints(3,2); imshow(newGBR,'XData',Yhat','YData',Xhat),axis on