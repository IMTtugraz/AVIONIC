function adjval = test_adjointness(K,Kh,imgdims,datadims)

xx = randn(imgdims);
yy = randn(datadims);

Kx = K(xx);
Khy = Kh(yy(:));

adjval = dot(yy(:),Kx(:)) - dot(Khy(:),xx(:));

display(['Adjoint value <K^H y, x> - <y, Kx> = ',num2str(adjval)]);

end