function z = my_fft(x,y) 


		% Bit-reverse
        n = size( x,2 );
        
%         n = lk(2);
        disp(n);
        m = log(n)/log(2);
%     disp(n);
%     m = 2;
		j = 1;
		n2 = n / 2;
        
		for i = 2:n - 1
         	n1 = n2;
			while (j >= n1) 
               
				j = j - n1;
				n1 = n1 / 2;
            end
			j = j + n1;

			if (i < j) 
				t1 = x(i);
				x(i) = x(j);
				x(j) = t1;
				t1 = y(i);
				y(i) = y(j);
				y(j) = t1;
            end
        end

		% FFT
		n1 = 0;
		n2 = 1;
 
		for i = 1: m
        
			n1 = n2;
			n2 = n2 + n2;
			a = 0;

			for j = 1:n1
             
                
%                 cos[i] = Math.cos(-2 * Math.PI * a / n);
% 			sin[i] = Math.sin(-2 * Math.PI * a / n);
                
				c = cos(-2 * pi * a / n);
				s = sin(-2 * pi * a / n);
				%a += 1 << (m - i - 1);
                
                a = a + bitshift(1,(m - i - 1));

% a =1;
    
				for k = j:n2:n
					t1 = c * x(k + n1) - s * y(k + n1);
					t2 = s * x(k + n1) + c * y(k + n1);
					x(k + n1) = x(k) - t1;
					y(k + n1) = y(k) - t2;
					x(k) = x(k) + t1;
					y(k) = y(k) + t2;
                
                      disp(x(k));
                   
                end
            end
         
        end
      
        z = x;
        %for i=1:size(x)
            
        %end
        
end