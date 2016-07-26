%Function to calculate the gradient of a [n,m] matrix using forward
%differences. The boundary extension is repetition of the last element
%
%(grad_u)_[i,j]=(u_[i+1,j]-u_[i,j],u_[i,j+1]-u_[i,j])

function [grad] = fgrad_3_1(u,timestep)

[n,m,o]=size(u);
t = 1/timestep;

grad = zeros(n,m,o,3);

%Set (1,0,0) derivatives
grad(1:n-1,:,:,1) =	u(2:n,:,:) - u(1:n-1,:,:);

%Set (0,1,0) derivatives
grad(:,1:m-1,:,2) =	u(:,2:m,:) - u(:,1:m-1,:);

%Set (0,0,1) derivatives
grad(:,:,1:o-1,3) = t*(	u(:,:,2:o) - u(:,:,1:o-1) );
