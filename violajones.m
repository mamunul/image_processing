
function [c,R] = haarfeature()
window = 24;
I = intimg(window);
features = [2,1;1,2;3,1;1,3;2,2];
c = 0;



[featureQuantity,n] = size(features);

R = zeros(162336,11,'uint32');
% str = zeros(163,5,'uint32');
for f=1:featureQuantity
    fw = features(f,1);
    fh = features(f,2);
    for width = fw:fw:(window+1)
        for height = fh:fh:(window+1)
            for x = 1:(window+1)-width
                for y = 1:(window+1)-height
                    c=c+1;
                    R(c,1) = x;
                    R(c,2) = y;
                    R(c,3) = width;
                    R(c,4) = height;
                    R(c,5) = f;
                    R(c,6) = width/fw;
                    R(c,7) = height/fh;
                    Area =  CheckForFeature(fw,fh,width,height,x,y,I);
                    
                    [mn dk] = size(Area);
                    R(c,8:(8+mn-1)) = Area(:);
                    
                end
            end
        end
    end
end
end

function Area = CheckForFeature(fw,fh,width,height,x,y,I)

Area = zeros(fw*fh,1);
%                     Area = zeros(4,1);
dx = width/fw;
dy = height/fh;
p = 1;
threshhold = 0.5;
for nx=x:dx:(x+width-dx)
    
    for ny=y:dy:(y+height-dy)
        
        A = 0; B = 0; C = 0; D = 0;
        
        A = I(nx-1+dx,ny-1+dy);
        if  ny >1
            C = I(nx-1+dx,ny-1);
        end
        if nx >1
            B = I(nx-1,ny-1+dy);
        end
        if nx >1 && ny >1
            D = I(nx-1,ny-1);
        end
        Area(p) = A - B - C  + D;
        
        p=p+1;
    end
    
end
end

function I = intimg(s)

i = imread('a.jpg');
i = imresize(i,[s s]);

imshow(i);
i = rgb2gray(i);

[m,n] = size(i);

I(1:m,1:n) = double (0.0);



for x = 1: m
    
    for y = 1:n
        
        a =0;b=0;c=0;
        
        if x>1
            a = I(x-1,y);
        end
        
        if y>1
            b = I(x,y-1);
        end
        
        if y>1 && x>1
            c = I(x-1,y-1);
        end
        
        I(x,y) = double (i(x,y)) + double(a) + double(b) - double(c);
        
    end
end
end



