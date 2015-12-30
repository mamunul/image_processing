

function re = identifyFace()

image = 'bc.jpg';

image = imread(image);

[s t r] = size(image);

original_s = s;

features = trainedFeatures('facedetection_last.xml');


faces = zeros(1,4);

% j = 1;
while s >= 24
    
    I = intimg(s, image);

    f = checkForWindow(features,I,original_s);
    
    if f ~= 0
    faces = cat(1,faces, f);
    end
    s = floor(s*.75);
end


end

function faceFrame = checkForWindow(features,I,original_s)

[p q] = size(I);

i=1;
faceFrame = 0;
for x = 1:p-24
    
    for y = 1:q-24
        pass = checkForClassifier(features,x,y,I);
        
        if  pass == 1
            
            f(1) = x * (original_s/p);
            f(2) = y * (original_s/p);
            f(3) = 24 * (original_s/p);
            f(4) = 24 * (original_s/p);
            
            faceFrame(i,1:4) = f;
            
            i = i +1;
        end
    end
end



end

function pass = checkForClassifier(features,img_x,img_y,I)

[m n] = size(features);

% classifier = [1,3,16,21,39,33,44,50,15,51,56,71,80,103,111,102,135,137,140,160,177,182,211,213];

classifier = [5,20,20,20];


pass = 1;

[p q] = size(classifier);

for k = 1:q
 
    if k> 1
       
        start_c =  end_c+1;
        end_c = start_c + classifier(k);
        
    else
        
        start_c = 1;
        end_c = classifier(k);
        
    end

    pass = 1;
    
      for i = start_c:end_c
        f1 = features(i,:);
        
        fw = f1(6);
        fh = f1(7);
        width = f1(8);
        height = f1(9);
        x = f1(4);
        y = f1(5);

        f = [img_x+x,img_y+y,width,height,fw,fh];
        
        theta = f1(2);
        polarity = f1(3);
        h = CheckForFeature(f,I,theta,polarity);
        
        if h ~= 1
   
            pass = 0;
            break;
        end
        
      end
    
      if pass == 0
          break;
      end
   
end

  
end

function feature = trainedFeatures(filename)

docNode = xmlread(filename);

face_detection_node = docNode.getDocumentElement;
entries = face_detection_node.getChildNodes;


list = entries.getElementsByTagName('feature');
feature_count = list.getLength;

feature = zeros(feature_count,9);

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
   feature(i,:) = [error,theta,polarity,x,y,fw,fh,w,h];
    
end


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
    
    if p == 2
        
        AN1 = abs(Area(1)-Area(2));
        AN2 = abs(Area(1)+Area(2));
        if AN2/AN1 <=1/(1+theta) && polarity == 1
            
            h(i) = 1;
        end
        
        if polarity == -1 && AN2/AN1 >=(1+theta)
            
            h(i) = 1;
        end
        
    elseif p == 3
        
        AN = Area(3)+Area(1);
        AN = AN/2;
        
        AN1 = abs(AN-Area(2));
        AN2 = abs(AN+Area(2));
        
        if AN2/AN1 <=1/(1+theta) && polarity == 1
            h(i) = 1;
        end
        if polarity == -1 && AN2/AN1 >=(1+theta)
            h(i) = 1;
        end
    elseif p == 4 
        
        AN1 = Area(1) + Area(4);
        AN2 = Area(2) + Area(3);
        
        if AN2/AN1 <=1/(1+theta) && polarity == 1
            h(i) = 1;
        end
        if polarity == -1 && AN2/AN1 >=(1+theta)
            h(i) = 1;
        end
        
    end
    
end

Area(p+1)=h(i)*1000000;

end
