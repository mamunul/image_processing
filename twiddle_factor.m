function t = twiddle_factor(N)


alpha = 2*pi/N;

S = sin(alpha);
C = 1 - 2*(sin(alpha/2).^2);

wcos(1) = 1;
wsin(1) = 0;

for K = 1:(N/8)-2
    
    wcos(K+1) = C*wcos(K) - S*wsin(K);
    wsin(K+1) = S*wcos(K) + C*wsin(K);
    
end


L = N/8;

wcos(L) = sqrt(2)/2;
wsin(L) = sqrt(2)/2;


for K= 0:(N/8)-1
    
    wcos(L+K) = wsin(L-K);
    wsin(L+K) = wcos(L-K);
    
end

L = N/4;
wcos(L) = 0;
wsin(L) = 1;

for K= 0:(N/4)
    
    wcos(L+K) = -wcos(L-K);
    wsin(L+K) = wsin(L-K);
    
end

end