
function P = viola_jones()

PositiveTrainingPath = 'faces';
NegativeTrainingPath = 'nonfaces';

starttime = generate_datetime();

trainig_face_limit = 100;

% trainig_face_limit = 4500;

[I1 m] = readImages(PositiveTrainingPath,'pgm',trainig_face_limit);

[I2 l] = readImages(NegativeTrainingPath,'pgm',trainig_face_limit);

I = cat(3,I1,I2);

y1 = ones(1,m);
y2 = zeros(1,l);
y = [y1,y2];

feature_count = 20000;
feature_list= haarfeature(24,feature_count);

[feature_count n] = size(feature_list);




i = 0;

feature_values=0;
for f_no =1:feature_count
    
    feature = feature_list(f_no,:);
    
    f_I = calculate_feature_value(I,feature,y);
    feature_values(1:(m+l),1:2,f_no) = f_I;
end


NumberOfClassifier = 50;
%     max = 3;
result = zeros(NumberOfClassifier,13);
w(1,1:m+l) = initialize_weight(m,l);
for t = 1:NumberOfClassifier
    
    w(t,1:m+l) = normalize_weight(w(t,:));
    
    [Tp Tm] = calculate_tpm1(w(t,:),y);
    
   
    
    err = 100;
    thresh = 0;
    polar = 0;
    for f_no =1:feature_count
        
       
        
        feature = feature_list(f_no,:);
        
        values = feature_values(:,:,f_no);
        
        fv_weight = fvplusweight(values,w(t,:));
        

        [err thresh polar] = bestfeature(feature,fv_weight,Tp,Tm,err,thresh,polar);

    end
  

    beta(t) = calc_beta(err);
    alpha(t) = calc_alpha(beta(t));
    error(t) = err;
    theta(t)=thresh;
    polarity(t)=polar;
    w(t+1,:) = calc_weight(w(t,:),beta(t),values,thresh,polar);
    
    display('classifier no------------');
    disp(t);

    value_no = t;
    value_x = double(feature(1));
    value_y = double(feature(2));
    value_fw = double(feature(5));
    value_fh = double(feature(6));
    value_w = double(feature(3));
    value_h = double(feature(4));
    
   
    result(t,:)=[t,t,err,thresh,alpha(t),beta(t) ,polar,value_x ,value_y,value_fw ,value_fh,value_w ,value_h];
    %     end
    
end


% C = sortrows(result,4);

% C=result(result(:,4) >30, :);
B = sortrows(result,3);
R = B(1:10000,:);

endtime = generate_datetime();
generateTrainingDb(R,starttime,endtime,trainig_face_limit,feature_count);

end

function fvw = fvplusweight(v,w)

[m n] = size(v);

for i = 1:m
    
    fvw(i,1)= v(i,1);
    fvw(i,2)= v(i,2);
    fvw(i,3)= w(i);
end


end

function beta = calc_beta(error)

beta = error/(1-error);

end

function [error thresh polarity] = bestfeature(feature,feature_value,tp,tm,error,thresh,polarity)
sp = 0;
sm = 0;
[m n]=size(feature_value);
feature_value = sortrows(feature_value,1);

for i = 1:m
    

    
    if feature_value(i,2) == 1
        sp = sp + feature_value(i,3);
    else
        sm = sm + feature_value(i,3);
    end
    
    v1 = sp+tm-sm;
    v2 = sm+tp-sp;
    
    e = min(v1,v2);
    
       
    if e<error
        error = e;
        thresh = feature_value(i,1);
        
        if e == v1
            polarity = -1;
        else
            polarity = 1;
        end
    end
end


end

function [tp tm]=calculate_tpm1(w,y)

tp = 0;
tm = 0;
[m n] = size(w);

for i = 1:n
if y(i) == 1
    tp = tp+ w(i);
 
else
    tm = tm + w(i);
  
end
end

end

function [tp tm]=calculate_tpm(w,y,values)

tp = 0;
tm = 0;
[m n] = size(w);
w_p = 0;w_n = 0;
for i = 1:n
if y(i) == 1
    tp = tp+ w(i)*values(i);
    w_p = w_p + w(i);
else
    tm = tm + w(i)*values(i);
    w_n = w_n + w(i);
end
end

tp = tp/w_p;
tm = tp/w_m;
end

function w = normalize_weight(wp)

[m n] = size(wp);
sum = 0;
for i = 1: n
    
    sum = sum + wp(i);
end


for i = 1: n
    
    w(i) = wp(i)/sum;
end

end

function dt = generate_datetime()

format shortg;
c = clock;

dt = [num2str(c(1)),'-',num2str(c(2)),'-',num2str(c(3)),' ',num2str(c(4)),':',num2str(c(5)),':',num2str(c(6))];


end

function s = generateTrainingDb(R,starttime,endtime,trainig_face_limit,feature_count)

docNode = com.mathworks.xml.XMLUtils.createDocument('face-detection');

entry_node = docNode.createElement('feature-list');
docNode.getDocumentElement.appendChild(entry_node);

[m n] = size(R);

sin11 = create_single_node(docNode,entry_node,'code-link','https://github.com/mamunul/image_processing.git');
docNode.getDocumentElement.appendChild(sin11);

sin12 = create_single_node(docNode,entry_node,'trainig-face-quantity',trainig_face_limit);
docNode.getDocumentElement.appendChild(sin12);

sin14 = create_single_node(docNode,entry_node,'trainig-nonface-quantity',trainig_face_limit);
docNode.getDocumentElement.appendChild(sin14);

sin13 = create_single_node(docNode,entry_node,'feature-count',feature_count);
docNode.getDocumentElement.appendChild(sin13);

sin1 = create_single_node(docNode,entry_node,'count',m);
docNode.getDocumentElement.appendChild(sin1);
%
sin2 = create_single_node(docNode,entry_node,'start-time',starttime);
docNode.getDocumentElement.appendChild(sin2);
%
sin3 = create_single_node(docNode,entry_node,'end-time',endtime);
docNode.getDocumentElement.appendChild(sin3);

for i = 1:m
    
    feature_node =  create_feature_node(docNode,entry_node,R(i,:));
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
        
        if m == 10
        skjfkj=0;
        end
        
        if m == limit
            break;
        end
    end
    
end
end

function nw = calc_weight(w,beta,values,thresh,p)

[a b] = size(w);
for i = 1:b
   
    if (p * values(i,1)) < (p * thresh)
        c = 1;
    else
        c = 0;
    end
    if values(i,2) == c
  
        nw(i) = w(i) * beta;
        
    else
        nw(i) = w(i);
    end
end




end

function alpha = calc_alpha(beta)


alpha =  log(1/beta);

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
    
    if h(i) == 1%should be 0
        
        if y(i) == 1
            Sp = Sp + w(i);
        else
            Sm = Sm + w(i);
        end
    end
end

error = min(Sp+Tm-Sm,Sm+Tp-Sp);

end

function feature_list = haarfeature(window,limit)


features = [2,1;1,2;3,1;1,3;2,2];
% features = [2,1;1,2;3,1;1,3];

[featureQuantity,n] = size(features);

% feature_list = zeros(141600,6,'uint32');
% feature_list = zeros(162336,6,'uint32');
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
                    
                    if t ==limit
                        break;
                    end
                end
                if t ==limit
                    break;
                end
            end
            if t ==limit
                break;
            end
        end
        if t ==limit
            break;
        end
    end
    if t ==limit
        break;
    end
    
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
    
    if abs(Ar) > 3
        jk=0;
    end
    f_value(i,1) = Ar;
    f_value(i,2) = label(i);
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
            im(i,j) = double(double(img(i,j)) - double(mean))/sd;
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