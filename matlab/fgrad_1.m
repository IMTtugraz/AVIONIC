%Function to calculate the component wise gradient of a [n,m,l] matrix using forward
%differences. The boundary extension is repetition of the last element
%
%(grad_u)_[i,j]=(u_[i+1,j]-u_[i,j],u_[i,j+1]-u_[i,j])

function [grad] = fgrad_1(u)

n=size(u,1);
m=size(u,2);
k = size(u,3);

grad = zeros(n,m,2,k);

%Set (1,0) derivatives
%grad(n,:,1) = 0;
grad(1:n-1,:,1,:) = u(2:n,:,:) - u(1:n-1,:,:);

%Set (0,1) derivatives
%grad(:,m,2) = 0;
grad(:,1:m-1,2,:) = u(:,2:m,:) - u(:,1:m-1,:);
