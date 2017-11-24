%% Kdata to Image, non-cartesian
function im = forward_opt_noncart_gpu( kdata, b1, w,FT)

[~,~,~,nframes] = size(kdata);
[n,m,ncoils] = size(b1);

im = zeros(n,m,nframes);

kdata = kdata.*permute(repmat(sqrt(w),[1 1 1 ncoils]),[1 2 4 3]);

for frame = 1:nframes
   
    im_ = zeros(n,m);
    for coil = 1:ncoils
         im_ =  im_ +  FT{frame}'*(col(squeeze(kdata(:,:,coil,frame)))).*conj(b1(:,:,coil));%prod(nufft_st{frame}.Kd)); 
    end
    im(:,:,frame) = im_;
end



