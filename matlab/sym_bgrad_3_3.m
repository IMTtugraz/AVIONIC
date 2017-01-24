%Function to calculate the symetric gradient of a [n,m,o,3] matrix using
%backward differences. The boundary extension is 0 for the 0 and n/m'th
%row/column


function [grad_v]=sym_bgrad_3_3(v,timestep)

n = size(v,1);
m = size(v,2);
o = size(v,3);
t = 1/timestep;

grad_v = zeros(n,m,o,6);

%%Diagonal entries--------------------------------------------------------

	grad_v(1,:,:,1)		= v(1    ,:,:,1);                      %Get (1,0,0) of v1
	grad_v(2:n-1,:,:,1)	= v(2:n-1,:,:,1) - v(1:n-2,:,:,1);
	grad_v(n,:,:,1)		=		 - v(n-1  ,:,:,1);

	grad_v(:,1    ,:,2)= v(:,1    ,:,2);                      %Get (0,1,0) of v2
	grad_v(:,2:m-1,:,2)= v(:,2:m-1,:,2) - v(:,1:m-2,:,2);
	grad_v(:,m    ,:,2)=		    - v(:,m-1  ,:,2);

	grad_v(:,:,1    ,3)= v(:,:,1    ,3);                      %Get (0,0,1) of v3
	grad_v(:,:,2:o-1,3)= v(:,:,2:o-1,3) - v(:,:,1:o-2,3);
	grad_v(:,:,o    ,3)=		    - v(:,:,o-1  ,3);
	grad_v(:,:,:,3) = grad_v(:,:,:,3)*t;	%Timestep-normalization

%%Off-diagonal entries----------------------------------------------------
tmp = zeros(n,m,o);

	%(1,2) entry
	tmp(:,1    ,:) =  v(:,1    ,:,1);                     	%Get (0,1,0) of v1
	tmp(:,2:m-1,:) =  v(:,2:m-1,:,1) - v(:,1:m-2,:,1);
	tmp(:,m    ,:) = 		 - v(:,m-1  ,:,1);

	grad_v(1,:    ,:,4)=tmp(1    ,:,:) + v(1    ,:,:,2);              		%Add (1,0,0) of v2
	grad_v(2:n-1,:,:,4)=tmp(2:n-1,:,:) + v(2:n-1,:,:,2) - v(1:n-2,:,:,2);
	grad_v(n,:    ,:,4)=tmp(n    ,:,:)     		    - v(n-1  ,:,:,2);

	grad_v(:,:,:,4)=grad_v(:,:,:,4)./2;                             	%devide by 2

	
	%(1,3) entry
	tmp(:,:,1    ) =  v(:,:,1    ,1);                     	%Get (0,0,1) of v1
	tmp(:,:,2:o-1) =  v(:,:,2:o-1,1) - v(:,:,1:o-2,1);
	tmp(:,:,o    ) =		 - v(:,:,o-1  ,1);
	tmp = tmp*t;	%Timestep normalization

	grad_v(1    ,:,:,5)=tmp(1    ,:,:) + v(1    ,:,:,3);              		%Add (1,0,0) of v3
	grad_v(2:n-1,:,:,5)=tmp(2:n-1,:,:) + v(2:n-1,:,:,3) - v(1:n-2,:,:,3);
	grad_v(n    ,:,:,5)=tmp(n    ,:,:)     		    - v(n-1  ,:,:,3);

	grad_v(:,:,:,5)=grad_v(:,:,:,5)./2;                             	%devide by 2


	%(2,3) entry
	tmp(:,:,1    ) =  v(:,:,1    ,2);                     	%Get (0,0,1) of v2
	tmp(:,:,2:o-1) =  v(:,:,2:o-1,2) - v(:,:,1:o-2,2);
	tmp(:,:,o    ) = 		 - v(:,:,o-1  ,2);
	tmp = tmp*t;	%Timestep normalization

	grad_v(:,1    ,:,6)=tmp(:,1    ,:) + v(:,1    ,:,3);              		%Add (0,1,0) of v3
	grad_v(:,2:m-1,:,6)=tmp(:,2:m-1,:) + v(:,2:m-1,:,3) - v(:,1:m-2,:,3);
	grad_v(:,m    ,:,6)=tmp(:,m    ,:) 		    - v(:,m-1  ,:,3);

	grad_v(:,:,:,6)=grad_v(:,:,:,6)./2;                             	%devide by 2
