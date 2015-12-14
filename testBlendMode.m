clear all
close all
clc

A = imread('a24.jpg');
% alpha(.5);
B = imread('a23.jpg');


imgd = im2double(A); 

r = imgd(:,:,1);
   r = 1 - r;


g = imgd(:,:,2);
  g = 1 - g;


b = imgd(:,:,3);
  b = 1 - b;

d = cat(3,r,g,b);


 
r2 = ghpf(r,12);
g2 = ghpf(g,12);
b2 = ghpf(b,12);
bj=.5;
e = cat(3,r2+bj,g2+bj,b2+bj);

% figure,imshow(A);
figure,imshow(e);

%  d = blendMode(A, e, 'overlay');
 

%   imshow(d,[]);
 
% imshow(A, 'InitialMag', 'fit')
%  % Make a truecolor all-green image.
% 
%  hold on 
%  h = imshow(e); 
%  hold off
%  set(h, 'AlphaData', .14)
