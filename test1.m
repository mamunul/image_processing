
J = imread('e.jpg');
figure;
imshow(J);
K = wiener2(J(:,:,1),[5 5]);

L = wiener2(J(:,:,2),[5 5]);

M = wiener2(J(:,:,3),[5 5]);

bo = cat(3, K, L, M);

b = imsharpen(bo,'Radius',1,'Amount',.2);

figure;
imshow(b);


%  inputImage = imnoise(imread('a.jpg'),'speckle',0.01);
%  outputImage1 = fcn(rgb2gray(J));
%  outputImage2 = fcn(rgb2gray(J),getnhood(strel('disk',1,0)));
%  figure,imshow(outputImage1)
%  figure,imshow(outputImage2)
 
 a = medfilt2(J(:,:,1),[3 3]);
 
 b = medfilt2(J(:,:,2),[3 3]);
 
 c = medfilt2(J(:,:,3),[3 3]);
 
 d = cat(3, a, b, c);
 
 a = ordfilt2(J(:,:,1),1,[1 1 1; 1 1 1; 1 1 1]);
 b = ordfilt2(J(:,:,2),1,[1 1 1; 1 1 1; 1 1 1]);
 c = ordfilt2(J(:,:,3),1,[1 1 1; 1 1 1; 1 1 1]);
 
 d = cat(3, a, b, c);
 
 figure;
 imshow(d);
 
 