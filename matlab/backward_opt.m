%% Image to Kdata
function kdata = backward_opt( img, mask, b1)

[n,m,nframes] = size(img);
ncoils = size(b1,3);


kdata = zeros(n,m,ncoils,nframes);

for frame = 1:nframes

	kdata(:,:,:,frame) = bsxfun(@times, ifft2( bsxfun(@times,b1,img(:,:,frame)) ) , mask(:,:,frame).*sqrt(n*m));
	
end



