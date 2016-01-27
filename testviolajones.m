

function re = identifyFace()



% image = 'testface/AJ_Lamas_0001.pgm';
image = 'testface/AJ_Cook_0001.pgm';
% image = 'testface/B1_00001.pgm';
image = imread(image);

image = imresize(image,[24  24]);

[s t r] = size(image);

original_s = s;

cascade = trainedFeatures('facedetection_last6.xml');


faces = 0;

% j = 1;
while s >= 24
    
%     imshow(image);
    I = intimg(s, image);

    f = checkForWindow(cascade,I,original_s,0.40);
    
    if f ~= 0
        [m n] = size(faces);
        [p q]= size(f);
        faces(m+1:m+p,1:4)=f;
        
        display(size(faces));
    end
    s = floor(s*.75); 
end


end

function faceFrame = checkForWindow(cascade,I,original_s,clissifier_threshold)

[p q] = size(I);

i=1;
faceFrame = 0;
for x = 1:(p+1)-24
    
    for y = 1:(q+1)-24
        r = checkForClassifier(cascade,x,y,I,clissifier_threshold);
%          
        if  r == 1
            
%             pass(i,1) = ppp;
           
          
            f(1) = x * (original_s/p);
            f(2) = y * (original_s/p);
            f(3) = 24 * (original_s/p);
            f(4) = 24 * (original_s/p);
%             f(5) = ppp;
            faceFrame(i,1:4) = f;
            
             i = i + 1;
%             
           
        end
    end
end



end

function r = checkForClassifier(cascade,img_x,img_y,I,classifier_threshold)

[m n p] = size(cascade);

% classifier = [1,3,16,21,39,33,44,50,15,51,56,71,80,103,111,102,135,137,140,160,177,182,211,213];


for i = 1:p
    
    single_cascade = cascade(:,:,i);
  

        r = detect_face(single_cascade,I,0);

    
        if r == 0 
            break;
        end
end

r
  
end


function r = detect_face(current_cascade,I,threshold)
[m n] = size(current_cascade);

current_cascade = sortrows(current_cascade,2);

result = 0;
for i = 1:m
    
    if current_cascade(i,1) == 0
        continue;
    end
    

    classifier = current_cascade(i,:);
    f(1) = classifier(6);
    f(2) = classifier(7);
    f(3) = classifier(10);
    f(4) = classifier(11);
    f(5) = classifier(8);
    f(6) = classifier(9);
    
    theta = classifier(4);
    alpha = classifier(3);
%     beta = classifier(6);
    polarity = classifier(5);
    
    classifier_threshold = current_cascade(m,2);
    
    f_v = calculate_feature_value(I,f,1);
    
    
    
    if polarity*f_v(1)  < polarity*theta  % confusion on polarity
        result = result + alpha;
        
    end
end

if result < classifier_threshold
    r = 0;
else
    r = 1;
end
end

function f_value = calculate_feature_value(Integral,f,label)
fw = f(5);
fh = f(6);
width = f(3);
height = f(4);
x = f(1);
y = f(2);
f_value = 0;
Area = zeros(fw*fh+1,1);
%                     Area = zeros(4,1);
dx = width/fw;
dy = height/fh;

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
    
    
    if p == 2
        
        AN1 = Area(1);
        AN2 = Area(2);
        Ar = AN1-AN2;
        
    elseif p == 3
        
        AN = Area(3)+Area(1);
        AN1 = AN/2;
        
        
        AN2 = Area(2);
        
        Ar = AN-AN2;
        
    elseif p == 4
        
        AN1 = Area(1) + Area(4);
        AN2 = Area(2) + Area(3);
        
        Ar = AN1-AN2;
    end
    
    
    f_value(i,1) = Ar;
    f_value(i,2) = label(i);
end


end


function cascade = trainedFeatures(filename)

docNode = xmlread(filename);

face_detection_node = docNode.getDocumentElement;
entries = face_detection_node.getChildNodes;

cascade_list = entries.getElementsByTagName('cascade');
cascade_count = cascade_list.getLength;

% cascade = zeros(cascade_count,20);

for j = 1:cascade_count
list = cascade_list.item(j-1).getElementsByTagName('feature');
feature_count = list.getLength;

feature = zeros(feature_count,11);

for i = 1:feature_count
    
  
   thisListItem = list.item(i-1);
   childNode = thisListItem.getFirstChild;
   

   while ~isempty(childNode)
      %Filter out text, comments, and processing instructions.
      if childNode.getNodeType == childNode.ELEMENT_NODE
         % Assume that each element has a single
         % org.w3c.dom.Text child.
         childText = char(childNode.getFirstChild.getData);

         nodeLabel = char(childNode.getTagName);
         switch nodeLabel
         case 'error';
           error = str2double(childText);
           case 'theta' ;
            theta = str2double(childText);
           case 'alpha' ;
            alpha = str2double(childText);
         case 'threshold' ;
            threshold = str2double(childText);
            case 'polarity' ;
            polarity = str2double(childText);
            case 'x' ;
            x = str2double(childText);
            case 'y' ;
            y = str2double(childText);
            case 'fw' ;
            fw = str2double(childText);
            case 'fh' ;
            fh = str2double(childText);
             case 'w' ;
            w = str2double(childText);
             case 'h' ;
            h = str2double(childText);
            
         end
      end  % End IF
      childNode = childNode.getNextSibling;
   end  % End WHILE
 
%    feature(i,1) = err;
   feature(i,:) = [error,threshold,alpha,theta,polarity,x,y,fw,fh,w,h];
    
end
n = 0
if j >1
    
[m n p] = size(cascade);
end

cascade(n+1:n+i,:,j) = feature;
end

% cc = cascade(:,:,2);
end

function I = intimg(s,i)

% i = imread('yale.jpg');
i = imresize(i,[s s]);

% imshow(i);
% i = rgb2gray(i);

[m,n,k] = size(i);

if(k == 3)
    i = rgb2gray(i);
    [m,n,k] = size(i);
end

i = normalize_image(i);

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

function im = normalize_image(img)


mean = get_mean(img);
sd = get_standard_deviation(img,mean);

[m n] = size(img);
sum = 0;


for i = 1:m
    for j = 1:n
        if sd == 0
            im(i,j) =  double(double(img(i,j)) - double(mean));
        else
            im(i,j) = double(double(img(i,j)) - double(mean))/(2*sd);
        end
    end
end


end

function mean = get_mean(img)

[m n] = size(img);
s = int32(0);

for i = 1:m
    for j = 1:n
        
        s = s + int32(img(i,j));
        
    end
end

mean = uint8(s/(m*n));

end

function sd = get_standard_deviation(img,mean)

sum = int32(0);
subtraction = 0;
[m n] = size(img);
for i = 1:m
    for j = 1:n
        
        a = img(i,j);
        b = double(a) - double(mean);
        subtraction = b;
        sum = sum + int32(subtraction)^2;
        
    end
end
sd = sum/int32(m*n);
sd = sqrt(double(sd));
end



function  h = CheckForFeature(f,Integral,theta,polarity)


fw = f(5);
fh = f(6);
width = f(3);
height = f(4);
x = f(1);
y = f(2);

Area = zeros(fw*fh+1,1);
%                     Area = zeros(4,1);
dx = width/fw;
dy = height/fh;

% threshhold = 0.5;
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
    
    diff = 0;
    
    if p == 2
        
        AN1 = Area(1);
        AN2 = Area(2);
        if AN2/AN1 <=1/(1+(theta*0.01)) && polarity == 1
            
            h(i) = 1;
            
        end
        
        if AN2/AN1 >=(1+(theta*0.01)) && polarity == -1
            
            h(i) = 1;
            
        end
        
    elseif p == 3
        
        AN = Area(3)+Area(1);
        AN1 = AN/2;
        
       
        AN2 = Area(2);
        
        if AN2/AN1 <=1/(1+(theta*0.01)) && polarity == 1
            h(i) = 1;
            
        end
        if AN2/AN1 >=(1+(theta*0.01)) && polarity == -1
            h(i) = 1;
            
        end
    elseif p == 4 
        
        AN1 = Area(1) + Area(4);
        AN2 = Area(2) + Area(3);
        
        if AN2/AN1 <=1/(1+(theta*0.01)) &&  polarity == 1
            h(i) = 1;
           
        end
        if AN2/AN1 >=(1+(theta*0.01)) && polarity == -1
            h(i) = 1;
            
        end
        
    end
    
%     diff
    
end

Area(p+1)=h(i)*1000000;

end
