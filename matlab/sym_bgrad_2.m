%Function to calculate the symetric gradient of a [n,m,2] matrix using
%backward differences. The boundary extension is 0 for the 0 and n/m'th
%row/column

%Input: n X n X 2 vector field representing the function
%Output: n X n X 3 dimensional vector field representing the gradient where
%the 3 coordinate stores equal non-diagonal values
%
%(grad_u)_[i,j]=(u1_[i,j]-u1_[i-1,j],u2_[i,j]-u2_[i,j-1],1/2*(u2_[i,j]-u2
%_[i-1,j]+u1_[i,j]-u1_[i,j-1]))
%

%Is -adjoint to fdiv_2 (tested)
function [grad_u]=sym_bgrad_2(u)

n = size(u,1);
m = size(u,2);
k = size(u,4);

grad_u = zeros(n,m,3,k);

grad_u(1,:,1,:)=u(1,:,1,:);                      %Get x derivative of u1
grad_u(2:n-1,:,1,:)=u(2:n-1,:,1,:)-u(1:n-2,:,1,:);
grad_u(n,:,1,:)=-u(n-1,:,1,:);

grad_u(:,1,2,:)=u(:,1,2,:);                      %Get y derivative of u2
grad_u(:,2:m-1,2,:)=u(:,2:m-1,2,:)-u(:,1:m-2,2,:);
grad_u(:,m,2,:)=-u(:,m-1,2,:);

%Calculate symetrized 3 part
grad_u(:,1,3,:)=u(:,1,1,:);                     %Set c to y derivative of u1
grad_u(:,2:m-1,3,:)=u(:,2:m-1,1,:)-u(:,1:m-2,1,:);
grad_u(:,m,3,:)=-u(:,m-1,1,:);

grad_u(1,:,3,:)=grad_u(1,:,3,:)+u(1,:,2,:);              %Add x derivative of u2
grad_u(2:n-1,:,3,:)=grad_u(2:n-1,:,3,:)+u(2:n-1,:,2,:)-u(1:n-2,:,2,:);
grad_u(n,:,3,:)=grad_u(n,:,3,:)-u(n-1,:,2,:);

grad_u(:,:,3,:)=grad_u(:,:,3,:)./2;                             %devide by 2
