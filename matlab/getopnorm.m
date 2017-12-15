function opnorm = getopnorm(K,Kh,imgdims)
% estimate ||K|| with power iterations

x1 = randn(imgdims);
y1 = Kh(K(x1));
for i=1:10
    if norm(y1(:))~=0
        x1 = y1./norm(y1(:));
    else
        x1 = y1;
    end
    [y1] = Kh(K(x1));
    l1 = y1(:)'*x1(:);
end
opnorm = max(abs(l1));

end