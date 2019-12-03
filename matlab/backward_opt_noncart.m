%% Image to Kdata, non-cartesian
function kdata = backward_opt_noncart( img, nufft_st, b1, dcf)


[n,m,~] = size(img);
[nsamplesonspoke,spokesperframe,nframes] = size(dcf); 

ncoils = size(b1,3);

kdata = zeros(nsamplesonspoke,spokesperframe,ncoils,nframes);

for frame=1:nframes,
    for coil = 1:ncoils
     kdata_ = img(:,:,frame).*b1(:,:,coil);   
     kdata(:,:,coil,frame) = reshape(...
        nufft(kdata_,nufft_st{frame})/sqrt(n*m),...(sqrt(prod(nufft_st{frame}.Kd))),...
        [nsamplesonspoke,spokesperframe]);%.*(w(:,:,frame));
    end
end

kdata = kdata.*permute(repmat(sqrt(dcf),[1 1 1 ncoils]),[1 2 4 3]);

