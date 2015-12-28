




function P = readTrainImage()

PositiveTrainingPath = 'faces';
NegativeTrainingPath = 'nonfaces';


starttime = generate_datetime();

limit = 10;
[I1 m] = readImages(PositiveTrainingPath,'pgm',limit);

[I2 l] = readImages(NegativeTrainingPath,'pgm',limit);

I = cat(3,I1,I2);

y1 = ones(1,m);
y2 = zeros(1,l);
y = [y1,y2]


feature_list= haarfeature(24);

[feature_count n] = size(feature_list);

% m = j;
% l = 0;
feature_count = 1000;
result = zeros(feature_count,13);

 i = 0;

for f_no = 1:feature_count
    
    feature = feature_list(f_no,:);
    
    lowest_err = 100;
    lowest_t = 0;
    
    
    w(1,1:m+l) = initialize_weight(m,l);
    
    for p = 1:2
        
    polarity = -1;
    if p == 2
        polarity = 1;
    end
    theta(1) = 0;
    max = 100;
    for t = 1:max
        
        if t >1
            theta(t) = theta(t-1)+1;
        end
        
        
        
        h(t,:) = CheckForFeature(feature,I,theta(t),polarity);
        epsilon(t) =calc_eps(h(t,:),y,w(t,:));
        alpha(t) = calc_alpha(epsilon(t));
        
        
        error(t) = calculate_error(h(t,:),y,w(t,:));
        
        w(t+1,:) = calc_weight(w(t,:),alpha(t),h(t,:),y);
        
        if(error(t) < lowest_err)
            lowest_err = error(t);
            lowest_t = t;
        end
        
        %             if error(t)>error(t-1)
        %
        %                 break;
        %             end
        
    end
    
    i = i+1;
%     display('feature no------------');
%     disp(f_no);
    %     display('feature------------');
    %     disp(feature);
    %     display('w------------');
    %     disp(w(lowest_t,:));
    %     display('theta------------');
    %     disp(theta(lowest_t));
    %     display('h------------');
    %     disp(h(lowest_t,:));
    %     display('alpha------------');
    %     disp(alpha(lowest_t));
    %     display('epsilon------------');
    %     disp(epsilon(lowest_t));
    %     display('error------------');
    %     disp(error(lowest_t));
    value_no = f_no;
    value_x = double(feature(1));
    value_y = double(feature(2));
    value_fw = double(feature(3));
    value_fh = double(feature(4));
    value_w = double(feature(5));
    value_h = double(feature(6));
    
   
    result(i,:)=[f_no,lowest_t,error(lowest_t),theta(lowest_t),alpha(lowest_t),epsilon(lowest_t),polarity,value_x ,value_y,value_fw ,value_fh,value_w ,value_h];
    
    end
end


B = sortrows(result,3);

R = B(1:feature_count*.01,:);

endtime = generate_datetime();
generateTrainingDb(R,starttime,endtime);



end


function dt = generate_datetime()

format shortg;
c = clock;

dt = [num2str(c(1)),'-',num2str(c(2)),'-',num2str(c(3)),' ',num2str(c(4)),':',num2str(c(5)),':',num2str(c(6))];


end

function s = generateTrainingDb(R,starttime,endtime)

docNode = com.mathworks.xml.XMLUtils.createDocument('face-detection');

entry_node = docNode.createElement('feature-list');
docNode.getDocumentElement.appendChild(entry_node);

[m n] = size(R);

sin1 = create_single_node(docNode,entry_node,'count',m);
docNode.getDocumentElement.appendChild(sin1);
% 
sin2 = create_single_node(docNode,entry_node,'start-time',starttime);
docNode.getDocumentElement.appendChild(sin2);
% 
sin3 = create_single_node(docNode,entry_node,'end-time',endtime);
docNode.getDocumentElement.appendChild(sin3);

for i = 1:m

  feature_node =  create_feature_node(docNode,entry_node,R(i,:))
  entry_node.appendChild(feature_node);
end

xmlFileName = ['facedetection','.xml'];
xmlwrite(xmlFileName,docNode);
type(xmlFileName);

end

function feature_node = create_feature_node(docNode,entry_node,f)

 rf_no = f(1);
 lowest_t = f(2);
 error = f(3);
 theta = f(4);
 alpha = f(5);
 epsilon = f(6);
 polarity = f(7);
 value_x = f(8);
 value_y = f(9);
 value_fw = f(10);
 value_fh = f(11);
 value_w = f(12);
 value_h = f(13);
 
 feature_node = docNode.createElement('feature');
 entry_node.appendChild(feature_node);
 
 sin1 = create_single_node(docNode,feature_node,'error',error);
 sin2 = create_single_node(docNode,feature_node,'theta',theta);
 
 sin4 = create_single_node(docNode,feature_node,'polarity',polarity);
sin5 = create_single_node(docNode,feature_node,'x',value_x);
sin6 = create_single_node(docNode,feature_node,'y',value_y);
sin7 = create_single_node(docNode,feature_node,'fw',value_fw);
sin8 = create_single_node(docNode,feature_node,'fh',value_fh);
sin9 = create_single_node(docNode,feature_node,'w',value_w);
sin10 = create_single_node(docNode,feature_node,'h',value_h);

 feature_node.appendChild(sin1);
 feature_node.appendChild(sin2);

 feature_node.appendChild(sin4);
 feature_node.appendChild(sin5);
 feature_node.appendChild(sin6);
 feature_node.appendChild(sin7);
 feature_node.appendChild(sin8);
 feature_node.appendChild(sin9);
 feature_node.appendChild(sin10);
    
end

function ct_node = create_single_node(docNode,entry_node,node,text)

ct_node = docNode.createElement(node);
count_text = docNode.createTextNode(num2str(text));
ct_node.appendChild(count_text);
% entry_node.appendChild(count_node);
end

function [I m] = readImages(path,extension,limit)
m = 0;
TrainFiles = dir(path);
for i = 1:size(TrainFiles,1)
    
    filename_compare_res = findstr(TrainFiles(i).name,extension);
    if isempty(filename_compare_res) ~=1
        df = strcat(path,'/', TrainFiles(i).name);
        image = imread(df);
        m=m+1;
        I(:,:,m) = intimg(24,image);
        
        if m == limit
            break;
        end
    end
    
end
end

function nw = calc_weight(w,alpha,h,y)

[a b] = size(w);
Z = 0;
for i = 1:b
    
    if h(i) == y(i)
        q(i) = exp(-alpha);
    else
        q(i) = exp(alpha);
    end
    
    Z = Z + w(i)*q(i);
end


for i = 1:b
    
    nw(i) = (w(i) * q(i))/ Z;
end


end

function alpha = calc_alpha(eps)


alpha = (1/2) * (log((1-eps)/eps));

end

function ep = calc_eps(h,yy,w)
ep = 0;
[m n] = size(yy);
for i = 1:n
    if(h(i) ~= yy(i))
        
        ep = ep+w(i);
    end
    
end

end
function w = initialize_weight(m,l)


for i = 1:m
    
    w(i) = 1/(2*m);
    
end

for i = m+1:l+m
    
    w(i) = 1/(2*l);
    
end
end
function error = calculate_error(h,y,w)

error = 1;

Tp = 0.0;
Tm = 0.0;
Sp = 0.0;
Sm = 0.0;

[m n] = size(y);

for i = 1:n
    
    if y(i) == 1
        Tp = Tp + w(i);
    else
        Tm = Tm + w(i);
    end
    
    if h(i) == 1
        
        if y(i) == 1
            Sp = Sp + w(i);
        else
            Sm = Sm + w(i);
        end
    end
end

error = min(Sp+Tm-Sm,Sm+Tp-Sp);

end

function feature_list = haarfeature(window)

features = [2,1;1,2;3,1;1,3];

[featureQuantity,n] = size(features);

feature_list = zeros(141600,6,'uint32');

t = 0;

for f=1:featureQuantity
    fw = features(f,1);
    fh = features(f,2);
    for width = fw:fw:(window + 1)
        for height = fh:fh:(window + 1)
            
            for pos_x = 1:(window+1) - width
                for pos_y = 1:(window+1) - height
                    t = t+1;
                    feature_list(t,1) = pos_x;
                    feature_list(t,2) = pos_y;
                    feature_list(t,3) = width;
                    feature_list(t,4) = height;
                    feature_list(t,5) = fw;
                    feature_list(t,6) = fh;
                end
            end
        end
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
        
    end
    
end

Area(p+1)=h(i)*1000000;

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



