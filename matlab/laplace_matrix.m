
function A = laplace_matrix(n,m,u,bcond);


A = sparse(n*m,n*m);
u = u(:);


%Inner diagonal cubes

	for j=1:n*m
		A(j,j) = 4; %Set diagonals
	
	end

	for j=1:n*m -1
		A(j+1,j) = -1;
		A(j,j+1) = -1;
	end

%Off diagonal cubes
	
	for j=1:n*(m-1)
		A(n+j,j) = -1;
		A(j,n+j) = -1;
	end

%Periodic boundary extension
if strcmp(bcond,'per')
	
	for j=n:n:n*m -1
		A(j+1,j) = 0;
		A(j,j+1) = 0;
	end
	
	
	for j=1:n %Top right and lower left block (periodic row)
		A(j,n*(m-1)+j) = -1;
		A(n*(m-1)+j,j) = -1;
	end
	
	for j=1:n:n*m %Peroidic column
		A(j,j+(n-1)) = -1;
		A(j+(n-1),j) = -1;
	end

%Symmetric boundary extension
elseif strcmp(bcond,'sym')


	for j=1:n:n*m - 1 %Set boundary of inner diagonal cubes
		A(j,j) = 3;
		A(j + n-1, j+n-1) = 3;
	end

	for j=1:n %Reduce first and last diagonal cube
		A(j,j) = A(j,j) - 1;
		A(n*(m - 1) + j,n*(m - 1) + j) = A(n*(m - 1) + j,n*(m - 1) + j) - 1;
	end



	for j=n:n:n*m -1
		A(j+1,j) = 0;
		A(j,j+1) = 0;
	end
	
end	


%Add diagonal entries

	for j=1:n*m
		A(j,j) = A(j,j) + u(j);
	end
