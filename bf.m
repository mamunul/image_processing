% input = imread('a.jpg');

function fout = f(image,i,j)

fout = impixel(image,i,j);
end

function wout = w(i,j,k,l,sigma_d,sigma_r)

c = (i-k)*(i-k)+(j-l)*(j-l);

a = -c/(2*sgma_d*sgma_d);


d = abs(f(i,j)-f(k,l))*(f(i,j)-f(k,l));

b = -d/(2*sgma_r*sgma_r);

wout = exp(a+b);

end

function r = nominator(i,j,k,l,sigma_d,sigma_r)

r = 0;
for k = 1:10
   
    for l = 1:10
        
        r = r + f(k,l) + w(i,j,k,l,sigma_d,sigma_r);
        
    end
end

end

function r = denominator(i,j,k,l,sigma_d,sigma_r)

for k = 1:10
   
    for l = 1:10
        
        r = r + wout(i,j,k,l,sigma_d,sigma_r);
        
    end
end

end

function gout = g(i,j,image,k,l,sigma_d,sigma_r)

gout  = nominator(i,j,k,l,image)/ denominator(i,j,k,l,image);

end


function bf = bf(image,sigma_d,sigma_r)

[x,y] = size(image);
for i = 1:x
   
    for j = 1:y
        out = g(i,j,image,sigma_d,sigma_r);
         
    end
end
end