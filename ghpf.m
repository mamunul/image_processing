% Code for input image (img) [300x300 array]

%  function g = ghpf(f,D0) 

 D0 = 12;
 A = imread('image.jpg'); 
% % alpha(.5);
 B = imread('a23.jpg');
% 
% 
 imgd = im2double(A); 
% 
 f = imgd(:,:,1);

[M,N]=size(f);
F=fft2(double(f));
% #define pi 3.14159265358979323846
% #define dB(x) ( ( (fabs(x)) > (0.0) ) ? (20.*log10(fabs(x))) : (-1000.) )
% #define Q 8
% float ar[Q], ai[Q];    /* array of points */
% float freq_test, mag;
% int i;

% fdata = fopen("data", "w");

% /* generate original signal */
%{
custom_fft(Q, ar, ai);

for i=0:Q
  mag = sqrt(ar(i)*ar(i)+ai(i)*ai(i));
end

%}
%{
  u=0:(M-1);
  v=0:(N-1);
  idx=find(u>M/2);
  u(idx)=u(idx)-M;
  idy=find(v>N/2);
  v(idy)=v(idy)-N;
  [V,U]=meshgrid(v,u);
  D=sqrt(U.^2+V.^2);
  H = 1 - exp(-(D.^2)./(2*(D0^2)));

%}

  for i=1:M
  for j=1:N
   a = floor(N/2) ;
   b = floor(M/2);
      if i==1 && j <=a
          distance(i,j)= j-1;
      elseif i==1 && j >a
          distance(i,j)= 1+a-(j-a);
       elseif(j==1 && i <=b)
          distance(i,j) =i-1;
       elseif(j==1 && i >b)
          distance(i,j) =1+b-(i-b);
       else 
                   w(i,j) = distance(i,1);
                   v(i,j) = distance(1,j);
          distance(i,j)=sqrt((w(i,j))^2+(v(i,j))^2);   
           
      end
      
     H(i,j)=1-exp(-(distance(i,j))^2/(2*(D0^2)));
     
     G(i,j) = H(i,j)*F(i,j);
%   distance=sqrt(U^2+V^2);
%   H(i,j)=1-exp(-(distance)^2/(2*(D0^2)));
  end
  end

%  G=H.*F;
g=real(ifft2(double(G)));
% subplot(1,2,1); 
% imshow(f); 
title('Input image'); 
% subplot(1,2,2); 
% imshow(g,[ ]); 
title('Enhanced image');
%  end


