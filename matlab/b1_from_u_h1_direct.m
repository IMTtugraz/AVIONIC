function b1 = b1_from_u_h1_direct(u,crec,mu)

[n,m,ncoils] = size(crec);

b1 = zeros(n,m,ncoils);

%Regularization par
if nargin < 3
    mu = 10^(-5);
end
		display(['Generating matrix... mu = ',num2str(mu)])
		A = laplace_matrix( n,m, mu*(abs(u).^2),'per');
		
		display('Going for coils...');
		
			for j=1:ncoils
					
				%Get right hand side
				v = mu.*conj(u).*crec(:,:,j);
	
				%Get phase of b1
				b1(:,:,j) = reshape( A \ v(:), n,m);
			end
			
	%Normalize
	b1_norm = sqrt( sum( abs(b1).^2 , 3 ) );
        for l=1:ncoils
            b1(:,:,l) = b1(:,:,l)./b1_norm;
        end
