%% Kdata to Image
function im = forward_opt( kdata, mask, b1)

[n,m,ncoils,nframes] = size(kdata);


im=zeros(n,m,nframes);

for coil=1:ncoils

        im = im + bsxfun(@times,fft2( squeeze(kdata(:,:,coil,:)).*mask  ),conj(b1(:,:,coil))./sqrt(n*m));

end




