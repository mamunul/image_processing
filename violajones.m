


function P = viola_jones()


PositiveTrainingPath = 'faces';
NegativeTrainingPath = 'nonfaces';

global bgImageArray
bgImageArray = readBgImages('nonfacecollection');


starttime = generate_datetime();
% trainig_face_limit = 500;

trainig_face_limit = 1000;

trainig_nonface_limit = 1000;

% getNewNegativeImage('/Users/mamunul/Downloads/Face Database/nonfacecollection');


[I1 m] = readPImages(PositiveTrainingPath,'pgm',trainig_face_limit);

I2 = 0;
feature_limit = -1;
feature_list= haarfeature(24,feature_limit);
f = 0.30; %false positive rate per cascade layer
d = 0.99; %detection rate per cascade layer
Ft = 0.0000006; %overall false positive rate
F(1) = 1;
D(1) = 1;
i = 1;
j = 0;
new_thresh = 0;
while F(i) > Ft || i < 10
    i = i+1;
    n = 10;
    F(i) = F(i-1);
    new_thresh = 0;
    
    [g h kl] = size(I2)
    [I22 l] = readNImages(bgImageArray,trainig_nonface_limit-kl);
    
    if kl >1
        
        I2 = cat(3,I2,I22);
    else
        I2 = I22;
    end
    
    [p q r]= size(I2);
      if r < trainig_nonface_limit
        break;
    end
    while F(i) > f*F(i-1)
        
        n = n+10;
        current_cascade = adaboost(I1,I2,feature_list,n);
        [p q] = size(current_cascade);
        for s = 1:p
            cascade(1:s,:) = current_cascade(1:s,:);
            new_thresh = 0;
            for r = 1:s
                new_thresh = new_thresh + cascade(r,5);
            end
            
            for r = 1:s
                cascade(r,2) = new_thresh;
            end
            
            [cf cd] = evaluate_cascade(cascade,I1,I2);
            
            F(i) = cf;
            D(i) = cd;
            new_thresh = decrease_threshold(d*D(i-1),cascade,I1);
            
            [p q] = size(cascade);
            
            for r = 1:p
                cascade(r,2) = new_thresh;
            end
            
            [cf cd] = evaluate_cascade(cascade,I1,I2);
            
            F(i) = cf;
            D(i) = cd;
            
            if F(i) <= f*F(i-1) && cd >= d*D(i-1)
                break;
            end
            
        end
    end
    
    
    n = 0
    if i >2
        
        [m n p] = size(whole_cascade);
    end
    
    [m p] = size(cascade);
    whole_cascade(n+1:n+m,:,i-1) = cascade;
    
    
    if F(i) > Ft
        
        I2 = evaluate_on_nonface(cascade,I2);
        
    end
    clear cascade;
    
    
  if i > 12
      break;
  end
    
end
endtime = generate_datetime();

generateTrainingDb(whole_cascade,starttime,endtime,trainig_face_limit,feature_limit);

end

function imageArray = getNewNegativeImage(path)

m = 0;
extension = '.jpg';
TrainFiles = dir(path);
for i = 1:size(TrainFiles,1)
    
    filename_compare_res = strfind(TrainFiles(i).name,extension);
    
    if isempty(filename_compare_res) ~=1
        df = strcat(path,'/', TrainFiles(i).name);
        image = imread(df);
        
        image = rgb2gray(image);
        m=m+1;
        
    end
    
end


end


function N_I2 = evaluate_on_nonface(current_cascade,I2)
[m p n] = size(I2);


j=0;
for i = 1:n
    
    cf = detect_face(current_cascade,I2(:,:,i),0);
    
    
    if cf == 1
        j = j +1;
        II = I2(:,:,i);
        
        N_I2(:,:,j) = II;
    end
end


end

function  best_threshold =decrease_threshold(required_d,current_cascade,I1)


[m p n] = size(I1);

[p q] = size(current_cascade);
classifier_threshold = 0;

for i = 1:p
    
    classifier_threshold = classifier_threshold + current_cascade(i,5);
end

min_thresh = 0;
max_thresh = classifier_threshold;
threshold = 0;
cd = 0;

bestcd = 0;
best_threshold = 0;

while (floor(max_thresh*1000) - floor(min_thresh*1000)) ~=0
    threshold = (max_thresh + min_thresh)/2;
    %     threshold = 0.8472;
    cd = 0;
    for i = 1:n
        
        cd = cd+ detect_face(current_cascade,I1(:,:,i),threshold);
        
    end
    cd = cd/n;
    
    
    if cd>bestcd
        bestcd = cd;
        best_threshold = threshold;
    end
    
    if cd == required_d
        break;
    end
    
    %     if floor(cd*100) < int8(required_d *100)
    %         max_thresh = threshold;
    %     elseif floor(cd*100) > int8(required_d *100)
    %         min_thresh = threshold;
    
    
    
    if (cd) < (required_d)
        max_thresh = threshold;
    elseif (cd) > (required_d)
        min_thresh = threshold;
    end
    
end
end

function [cf cd] = evaluate_cascade(current_cascade,I1,I2)

[m p n] = size(I1);
cd = 0;
cf = 0;
for i = 1:n
    
    cd = cd + detect_face(current_cascade,I1(:,:,i),0);
    
end

cd = cd/n;

[m q n] = size(I2);

for i = 1:n
    
    cf = cf + detect_face(current_cascade,I2(:,:,i),0);
    
end

cf = cf/n;

end

function r = detect_face(current_cascade,I,threshold)
[m n] = size(current_cascade);

result = 0;
for i = 1:m
    
    classifier = current_cascade(i,:);
    f(1) = classifier(8);
    f(2) = classifier(9);
    f(3) = classifier(12);
    f(4) = classifier(13);
    f(5) = classifier(10);
    f(6) = classifier(11);
    
    theta = classifier(4);
    alpha = classifier(5);
    beta = classifier(6);
    polarity = classifier(7);
    
    classifier_threshold = current_cascade(m,2);
    
    f_v = calculate_feature_value(I,f,1);
    
    
    
    if polarity*f_v(1)  < polarity*theta  % confusion on polarity
        result = result + alpha;
        
    end
end

if threshold ~= 0
    classifier_threshold = threshold;
end

if result < classifier_threshold
    r = 0;
else
    r = 1;
end
end

function result = adaboost(I1,I2,feature_list,NumberOfClassifier)

[dgd jk m] = size(I1);
[ertt jk l] = size(I2);

I = cat(3,I1,I2);

y1 = ones(1,m);
y2 = zeros(1,l);
y = [y1,y2];

[feature_count n] = size(feature_list);

i = 0;

feature_values=0;
for f_no =1:feature_count
    
    feature = feature_list(f_no,:);
    
    f_I = calculate_feature_value(I,feature,y);
    feature_values(1:(m+l),1:2,f_no) = f_I;
end

result = zeros(NumberOfClassifier,14);
w(1,1:m+l) = initialize_weight(m,l);

classifier_threshold = 0;
for t = 1:NumberOfClassifier
    
    w(t,1:m+l) = normalize_weight(w(t,:));
    
    [Tp Tm] = calculate_tpm1(w(t,:),y);
    
    err = 100;
    thresh = 0;
    polar = 0;
    s_f_no = 0;
    
    
    for f_no =1:feature_count
        
        feature = feature_list(f_no,:);
        
        values = feature_values(:,:,f_no);
        
        fv_weight = fvplusweight(values,w(t,:));
        
        [err thresh polar] = bestfeature(feature,fv_weight,Tp,Tm,err);
        
        if thresh ~= -100
            beta(t) = calc_beta(err);
            alpha(t) = calc_alpha(beta(t));
            error(t) = err;
            theta(t)=thresh;
            polarity(t)=polar;
            best_feature = feature;
            best_values = values;
        end
        
    end
    
    w(t+1,:) = calc_weight(w(t,:),beta(t),best_values,theta(t),polarity(t));
    
    display('classifier no------------');
    disp(t);
    
    value_no = t;
    value_x = double(best_feature(1));
    value_y = double(best_feature(2));
    value_fw = double(best_feature(5));
    value_fh = double(best_feature(6));
    value_w = double(best_feature(3));
    value_h = double(best_feature(4));
    
    
    classifier_threshold = classifier_threshold + alpha(t);
    
    result(t,:)=[t,classifier_threshold,err,theta(t),alpha(t),beta(t) ,polarity(t),value_x ,value_y,value_fw ,value_fh,value_w ,value_h,l];
    
    
end

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

function [error thresh polarity] = bestfeature(feature,feature_value,tp,tm,error)
sp = 0;
sm = 0;
[m n]=size(feature_value);
feature_value = sortrows(feature_value,1);
thresh = -100;
polarity = -100;
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

function s = generateTrainingDb(whole_cascade,starttime,endtime,trainig_face_limit,feature_count)

docNode = com.mathworks.xml.XMLUtils.createDocument('face-detection');

entry_node = docNode.createElement('feature-list');
docNode.getDocumentElement.appendChild(entry_node);

[m n p] = size(whole_cascade);

sin11 = create_single_node(docNode,entry_node,'code-link','https://github.com/mamunul/image_processing.git');
docNode.getDocumentElement.appendChild(sin11);

sin12 = create_single_node(docNode,entry_node,'trainig-face-quantity',trainig_face_limit);
docNode.getDocumentElement.appendChild(sin12);

sin14 = create_single_node(docNode,entry_node,'trainig-nonface-quantity',trainig_face_limit);
docNode.getDocumentElement.appendChild(sin14);

sin13 = create_single_node(docNode,entry_node,'feature-count',feature_count);
docNode.getDocumentElement.appendChild(sin13);


%
sin2 = create_single_node(docNode,entry_node,'start-time',starttime);
docNode.getDocumentElement.appendChild(sin2);
%
sin3 = create_single_node(docNode,entry_node,'end-time',endtime);
docNode.getDocumentElement.appendChild(sin3);


for i = 1:p
    ocascade = whole_cascade(:,:,i);
    
    [k l m] = size(ocascade);
    l = 0;
    for j = 1:k
        if ocascade(j,1) ~=0
            l = l +1;
            cascade(l,:) = ocascade(j,:);
            
        end
    end
    
    
    
    
    cascade_node  = docNode.createElement('cascade');
    
    classifier_node  = docNode.createElement('classifier');
    
    for j = 1:l
        
        feature_node =  create_feature_node(docNode,classifier_node,cascade(j,:));
        classifier_node.appendChild(feature_node);
    end
    
    %     cascade(j,:) = cascade(m,2);
    threshold = cascade(l,2);
    training_nonface_count = cascade(l,14);
    threshold_node = create_single_node(docNode,cascade_node,'threshold',threshold);
    count_node = create_single_node(docNode,cascade_node,'count',l);
    nonface_count_node = create_single_node(docNode,cascade_node,'nonface_count',training_nonface_count);
    
    cascade_node.appendChild(classifier_node);
    %     cascade_node.appendChild(feature_node);
    
    cascade_node.appendChild(nonface_count_node);
    cascade_node.appendChild(threshold_node);
    cascade_node.appendChild(count_node);
    entry_node.appendChild(cascade_node);
end

sin1 = create_single_node(docNode,entry_node,'count',p);
docNode.getDocumentElement.appendChild(sin1);
xmlFileName = ['facedetection','.xml'];
xmlwrite(xmlFileName,docNode);
type(xmlFileName);

end

function feature_node = create_feature_node(docNode,entry_node,f)

rf_no = f(1);
threshold = f(2);
error = f(3);
theta = f(4);
alpha = f(5);
beta = f(6);
polarity = f(7);
value_x = f(8);
value_y = f(9);
value_fw = f(10);
value_fh = f(11);
value_w = f(12);
value_h = f(13);

feature_node = docNode.createElement('feature');
entry_node.appendChild(feature_node);

sin0 = create_single_node(docNode,feature_node,'threshold',threshold);
sin1 = create_single_node(docNode,feature_node,'error',error);
sin2 = create_single_node(docNode,feature_node,'theta',theta);
sin3 = create_single_node(docNode,feature_node,'alpha',alpha);
sin4 = create_single_node(docNode,feature_node,'polarity',polarity);
sin5 = create_single_node(docNode,feature_node,'x',value_x);
sin6 = create_single_node(docNode,feature_node,'y',value_y);
sin7 = create_single_node(docNode,feature_node,'fw',value_fw);
sin8 = create_single_node(docNode,feature_node,'fh',value_fh);
sin9 = create_single_node(docNode,feature_node,'w',value_w);
sin10 = create_single_node(docNode,feature_node,'h',value_h);

% feature_node.appendChild(sin0);
feature_node.appendChild(sin1);
feature_node.appendChild(sin2);
feature_node.appendChild(sin3);
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

function [I m] = readPImages(path,extension,limit)
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

function [I2 m] = readNImages(bgArray,limit)

global bgImageArray
[m n] = size(bgArray);
I2 = 0;
for i = 1:m
    
    path = bgArray(i);
    
    bgImage = imread(char(path));
    
    [w h c] = size(bgImage);
    
    bgImage = imresize(bgImage,[ceil(w/2) ceil(h/2)]);
    
    imgArray = scrollImage(bgImage);
   [d e f] = size(I2);
    if f > 1
        I2 = cat(3,I2,imgArray);
    else
        I2 = imgArray;
    end
    bgImageArray(i) = '';
    
    [p q r] = size(I2);
    
    if limit < r
        break;
    else
        
    end
end

end

function imgArray = scrollImage(image)

[imgh imgw c] = size(image);

width = 80;
height = 80;
k = 1;

for  i = 1: ceil(height/3) : (imgw - height)
    
    for  j = 1: ceil(width/3) : (imgh - width)
        
        
        % 		imshow(image);
        
        crop = imcrop(image,[i j width height]);
        %                 imshow(crop);
        crop = rgb2gray(crop);
        
        [m n] = size(crop);
        
        if m ~= 81 || n ~= 81 
            sbc = 0;
        end
        
        
        
       crop = imresize(crop,[24 24]);
        imgArray(:,:,k) = crop(:,:);
        k = k+1;
    end
    
    
end





end

function arr = readBgImages(path)
limit = 190; extension = 'jpg';
TrainFiles = dir(path);
m = 0;
%  arr(1,:) = 'sfsdf';
for i = 1:size(TrainFiles,1)
    
    filename_compare_res = findstr(TrainFiles(i).name,extension);
    if isempty(filename_compare_res) ~=1
        df = strcat(path,'/', TrainFiles(i).name);
        m=m+1;
        arr(m,:) = cellstr(df);
        
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
                    
                    if t == limit
                        break;
                    end
                end
                if t == limit
                    break;
                end
            end
            if t == limit
                break;
            end
        end
        if t == limit
            break;
        end
    end
    if t == limit
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
