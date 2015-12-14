i = imread('b.jpg');

kl = image_denoise_rgb_3x3(i);

% kl = image_denoise_rgb_3x3(kl);
% 
% kl = image_denoise_rgb_3x3(kl);

c = [1 5 1,5 25 5, 1 5 1]/40;

% c = [1 2 1,2 4 2, 1 2 1]/16;

f = imfilter(i,c,'conv');

j = imadjust(i,[.1 .1 0; .2 .3 1],[]);

% k = edge(i,'canny');

salt = imnoise(i,'salt & pepper',0.02);

% ff = fft2(i)
% ff = fftshift(ff);
%  ff = mat2gray(ff);

is = imguidedfilter(i);

can = edge(rgb2gray(i),'Sobel');

subplot(1,2,1), imshow(f, 'InitialMagnification', 250);
subplot(1,2,2), imshow(can, 'InitialMagnification', 250);
% subplot(1,3,3), imshow(kl);
title('Process');
