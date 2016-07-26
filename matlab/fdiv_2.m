%Function to calculate the divergence of a [n,m,3] dimensional matrix using
%forward differences. The boundary extension is repetition of the last
%element.
%Input: n X n X 3 vector field representing the function (3rd value for
%non-diagonal values)
%Output: n X n X 2 dimensional vector field representing divergence
%
%!Remember! grad*=-div i.e. -div is dual operator of gradient
%
%(grad_u)_[i,j]=(u1_[i,j]-u1_[i-1,j],u2_[i,j]-u2_[i,j-1],1/2*(u2_[i,j]-u2
%_[i-1,j]+u1_[i,j]-u1_[i,j-1]))
%

% Is -adjoint to sym_bgrad_1 (tested)
function [div_v]=fdiv_2(v)

n = size(v,1);
m = size(v,2);
k = size(v,4);

div_v = zeros(n,m,2,k);

div_v(1:n-1,:,1,:)=reshape( v(2:n,:,1,:)-v(1:n-1,:,1,:)		,n-1,m,1,k);
div_v(:,1:m-1,1,:)=div_v(:,1:m-1,1,:)+reshape( v(:,2:m,3,:)-v(:,1:m-1,3,:)	,n,m-1,1,k);

div_v(1:n-1,:,2,:)=reshape(v(2:n,:,3,:)-v(1:n-1,:,3,:)	,n-1,m,1,k);
div_v(:,1:m-1,2,:)=div_v(:,1:m-1,2,:)+reshape( v(:,2:m,2,:)-v(:,1:m-1,2,:)	,n,m-1,1,k);



