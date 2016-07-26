%Function to calculate the divergence of a [n,m,3] dimensional matrix using
%forward differences. The boundary extension is repetition of the last
%element.



function [div_w]=d3_fdiv_2(w,timestep)

n = size(w,1);
m = size(w,2);
o = size(w,3);
t = 1/timestep;

div_w = zeros(n,m,o,3);
tmp = zeros(n,m,o);	
	
%First row
tmp(1:n-1,:    ,:)    =				    w(2:n,:,:,1)-w(1:n-1,:,:,1);	%(1,0,0) of w1
tmp(:    ,1:m-1,:)    = tmp(:,1:m-1,:) + 	    w(:,2:m,:,4)-w(:,1:m-1,:,4);	%(0,1,0) of w4
tmp(:    ,:,1:o-1)    = tmp(:,:,1:o-1) + 	t*( w(:,:,2:o,5)-w(:,:,1:o-1,5) );	%(0,0,1) of w5
div_w(:,:,:,1) = tmp;
tmp = zeros(n,m,o);	
	
%Second row
tmp(1:n-1,:    ,:)    =				    w(2:n,:,:,4)-w(1:n-1,:,:,4);	%(1,0,0) of w4
tmp(:    ,1:m-1,:)    = tmp(:,1:m-1,:) + 	    w(:,2:m,:,2)-w(:,1:m-1,:,2);	%(0,1,0) of w2
tmp(:    ,:,1:o-1)    = tmp(:,:,1:o-1) + 	t*( w(:,:,2:o,6)-w(:,:,1:o-1,6) );	%(0,0,1) of w6
div_w(:,:,:,2) = tmp;
tmp = zeros(n,m,o);	

%Third row
tmp(1:n-1,:    ,:)    =				    w(2:n,:,:,5)-w(1:n-1,:,:,5);	%(1,0,0) of w5
tmp(:    ,1:m-1,:)    = tmp(:,1:m-1,:) + 	    w(:,2:m,:,6)-w(:,1:m-1,:,6);	%(0,1,0) of w6
tmp(:    ,:,1:o-1)    = tmp(:,:,1:o-1) + 	t*( w(:,:,2:o,3)-w(:,:,1:o-1,3) );	%(0,0,1) of w3
div_w(:,:,:,3) = tmp;






