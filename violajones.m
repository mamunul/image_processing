

function P = readTrainImage()

PositiveTrainingPath = 'faces';
TrainFiles = dir(PositiveTrainingPath);

field_feature = 'feature';
field_y = 'y';

field_w = 'w';
field_ep = 'ep';
field_B = 'B';
field_a = 'a';
field_h = 'h';

j=0;
for i = 1:size(TrainFiles,1)
    
    filename_compare_res = findstr(TrainFiles(i).name,'pgm');
    if isempty(filename_compare_res) ~=1
        df = strcat(PositiveTrainingPath,'/', TrainFiles(i).name);
        image = imread(df);
         j=j+1;
        I(:,:,j) = intimg(24,image);
        
        if j == 2
            break;
        end
    end
    
end
[c, R, h]= haarfeature(24,I,j);

value_y = 1;
value_w = i;
vale_ep = i;
value_B = i;
value_a = i;
value_h = i;

feature = struct(field_w,value_w,field_ep,vale_ep,field_B,value_B,field_a,value_a,field_h,value_h);

result(i) = struct(field_feature,feature,field_y,value_y);



end

function [t, R, h] = haarfeature(window,I,num_of_positives)
% window = 24;
% I = intimg(window,image);
% features = [2,1;1,2;3,1;1,3;2,2];
features = [2,1;1,2;3,1;1,3];




[featureQuantity,n] = size(features);

% R = zeros(162336,12,'uint32');
R = zeros(141600,12,'uint32');
% h = zeros(141600,2,'uint32');
% str = zeros(163,5,'uint32');
t = 0;
w = zeros(141600,num_of_positives,'uint32');
for f=1:featureQuantity
    fw = features(f,1);
    fh = features(f,2);
    for width = fw:fw:(window + 1)
        for height = fh:fh:(window + 1)
            
            for pos_x = 1:(window+1) - width
                for pos_y = 1:(window+1) - height
                    t = t+1;
                    R(t,1) = pos_x;
                    R(t,2) = pos_y;
                    R(t,3) = width;
                    R(t,4) = height;
                    R(t,5) = f;
                    R(t,6) = width/fw;
                    R(t,7) = height/fh;
                    [Area, hs] =  CheckForFeature(fw,fh,width,height,pos_x,pos_y,I);
                    
                    [mn, dk] = size(Area);
                    R(t,8:(8+mn-1)) = Area(:);
                    h(t,1:num_of_positives) = hs;
                    w(1,t) = 1/(2*num_of_positives);
%                     epsilon = 100;
%                     beta = (epsilon/(1-epsilon));
%                     alpha = log(1/beta);
                end
            end
        end
    end
end
end

function [Area, h] = CheckForFeature(fw,fh,width,height,x,y,Integral)

Area = zeros(fw*fh+1,1);
%                     Area = zeros(4,1);
dx = width/fw;
dy = height/fh;

threshhold = 0.5;
[asf kjh number_of_image]=size(Integral);

for i=1:number_of_image

    I = Integral(:,:,i);
    p = 0;
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
        p=p+1;
        Area(p) = A - B - C  + D;
        
        
    end
    
end

h(i) = 0;

if p == 2
    
    AN1 = abs(Area(1)-Area(2));
    AN2 = abs(Area(1)+Area(2));
    if AN2/AN1 <=3
        
        h(i) = 1;
    end
    
    
    
elseif p == 3
    
    AN = Area(3)+Area(1);
    AN = AN/2;
    
    AN1 = abs(AN-Area(2));
    AN2 = abs(AN+Area(2));
    
    if AN2/AN1 <=3
        
        h(i) = 1;
    end
    
    
end

end

Area(p+1)=h(i)*1000000;

end

function I = intimg(s,i)

% i = imread('yale.jpg');
i = imresize(i,[s s]);

% imshow(i);
% i = rgb2gray(i);

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



