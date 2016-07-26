function [y] = sp_div_3_3(x,Dx,Dy,Dz,ds,dt)

[n,m,t,k] = size(x);
N = n*m*t;

y =  - reshape( ( Dx'*reshape(x(:,:,:,1),N,1) + Dy'*reshape(x(:,:,:,2),N,1)  	)/ds + ...
		( Dz'*reshape(x(:,:,:,3),N,1)  					)/dt ,n,m,t );
		
