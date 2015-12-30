

function re = identifyFace()

image = 'bc.jpg';

image = imread(image);

[s t r] = size(image);

original_s = s;

features = trainedFeatures('facedetection_last.xml');


% faces = zeros(10,6);

j = 1;
while s >= 24
    
    I = intimg(s, image);

    f = checkForWindow(features,I,original_s);
    
    faces = [faces f];
    
    s = floor(s*.75);
end


end

function faceFrame = checkForWindow(features,I,original_s)

[p q] = size(I);

i=1;

for x = 1:p
    
    for y = 1:q
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

classifier = [1,5,10,10,10,10,15,15,15,20,20,20,25,25];

classifier = [1,5,20,20,20];


    pass = 1;

for k = 1:size(classifier)
    
    start_c = classifier(k);
    end_c = classifier(k+1);
    
    if k> 1
       
        
        
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

%         f = [x,y,width,height,fw,fh];
        f = [img_x+x,img_y+y,width,height,fw,fh];
        
        theta = f1(2);
        polarity = f1(3);
        h = CheckForFeature(f,I,theta,polarity);
        
        if h ~= 1
            
         
                       
          
            j = j+1;
            
            pass = 0;
            break;
        end
        
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
