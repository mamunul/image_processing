function f2 = my_2dfft(in)
   [M_2P,N_2P] = size(in);
	for m = 1:M_2P 
			for n = 1:N_2P
				v(n) = in(m,n);
            end
			vf = my_fft(v); 
			for n = 1:N_2P
				f2(m,n) = vf(n);
            end
    end
		

% 		// transform second dimension

		for n = 1:N_2P
			for m = 1:M_2P
				v(m) = in(m,n);
            end
			vf = my_fft(v); 
			for m = 1:M_2P
				f2(m,n) = vf(m);
            end
        end
		


end