%Function to calculate the divergence of a [n,m]-matrix using backward
%differences. The boundary extension mode is 0 for the 0' and n/m'th
%row/colomn
%
%!Remember! grad*=-div i.e. -div is dual operator of gradient
%
%div(v)_[i,j]=v_[i,j]-v_[i-1,j] + v_[i,j]-v_[i,j-1]

function [div_v] = bdiv_3_3(v,timestep)

n = size(v,1);
m = size(v,2);
o = size(v,3);
t = 1/timestep;

div_v = zeros(n,m,o);
tmp = zeros(n,m,o);

	%(1,0,0) of v1
	tmp(1    ,:,:) =  v(1    ,:,:,1);
	tmp(n    ,:,:) = -v(n-1  ,:,:,1);
	tmp(2:n-1,:,:) =  v(2:n-1,:,:,1) - v(1:n-2,:,:,1);

	%(0,1,0) of v2
	tmp(:,1    ,:) = tmp(:,1    ,:) + v(:,1    ,:,2);
	tmp(:,m    ,:) = tmp(:,m    ,:) - v(:,m-1  ,:,2);
	tmp(:,2:m-1,:) = tmp(:,2:m-1,:) + v(:,2:m-1,:,2) - v(:,1:m-2,:,2);
	
	%(0,0,1) of v3
	div_v(:,:,1    ) = tmp(:,:,1    ) + t*( v(:,:,1    ,3) );
	div_v(:,:,o    ) = tmp(:,:,o    ) - t*( v(:,:,o-1  ,3) );
	div_v(:,:,2:o-1) = tmp(:,:,2:o-1) + t*( v(:,:,2:o-1,3) - v(:,:,1:o-2,3) );
	
	
	
