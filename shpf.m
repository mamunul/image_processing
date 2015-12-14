function result= shpf(f) 

    dt = 5;
	RC = 6.0;

	a = (RC / (RC + dt));
    
    r=f(:,:,1);
    g=f(:,:,2);
    b=f(:,:,3);
    
    r1=r;
    g1=g;
    b1=b;
    
    for x = 1:size(r)
        if  x ~= 1  
            r1(x) = a * (r1(x-1) + r(x) - r(x-1));
            g1(x) = a * (g1(x-1) + g(x) - g(x-1));
            b1(x) = a * (b1(x-1) + b(x) - b(x-1));
        else
            r1(x) = r(x);
            g1(x) = g(x);
            b1(x) = b(x);
        end
    end
    
    result = cat(3,r1,g1,b1);

end