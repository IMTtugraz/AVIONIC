function [y] = sp_grad_3_1(x,Dx,Dy,Dz,ds,dt)

[n,m,t] = size(x);

y = reshape( [ (Dx*x(:))/ds ; (Dy*x(:))/ds ; (Dz*x(:))/dt] , n,m,t,3);
