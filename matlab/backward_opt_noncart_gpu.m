%% Image to Kdata, non-cartesian
function kdata = backward_opt_noncart_gpu( img, b1, dcf, FT)


[n,m,~] = size(img);
[nsamplesonspoke,spokesperframe,nframes] = size(dcf); 

ncoils = size(b1,3);

kdata = zeros(nsamplesonspoke,spokesperframe,ncoils,nframes);

for frame=1:nframes,
    for coil = 1:ncoils
     kdata_ = img(:,:,frame).*b1(:,:,coil);   
     kdata(:,:,coil,frame) = reshape(...
        FT{frame}*(kdata_),...
        [nsamplesonspoke,spokesperframe]);%.*(w(:,:,frame));
    end
end

kdata = kdata.*permute(repmat(sqrt(dcf),[1 1 1 ncoils]),[1 2 4 3]);

