function [ recon ] = linear_recon( data, mask, b1 )

    [n,m,ncoils,nframes] = size(data);

    crec = zeros(n,m,ncoils);
    for j = 1:ncoils
        data_ = sum( squeeze(data(:,:,j,:)), 3);
        msum = sum(mask,3);
        data_(msum>0) = data_(msum>0)./msum(msum>0);
        crec(:,:,j) = fft2( data_ )./sqrt(n*m);
    end
    u = sqrt( sum( abs(crec).^2 , 3) );
    datanorm = 255./median(u(u>=0.9.*max(u(:))));

    u0 = sum(crec.*datanorm.*conj(b1),3);
    clear u crec
    data = data.*datanorm;      


    rhs = forward_opt(data,mask,b1);
    rhs = rhs(:);

    SM = @(x) col(forward_opt(backward_opt(reshape(x,[n,m,nframes]),mask,b1),mask,b1));

    recon = pcg(SM,rhs,1e-12,5,[],[],col(repmat(u0,[ 1 1 nframes])));

    recon = reshape(recon,[n,m,nframes])./datanorm;

end

function im = forward_opt( kdata, mask, b1)

    [n,m,ncoils,nframes] = size(kdata);


    im=zeros(n,m,nframes);

    for coil=1:ncoils

            im = im + bsxfun(@times,fft2( squeeze(kdata(:,:,coil,:)).*mask  ),conj(b1(:,:,coil))./sqrt(n*m));

    end

end


function kdata = backward_opt( img, mask, b1)

    [n,m,nframes] = size(img);
    ncoils = size(b1,3);


    kdata = zeros(n,m,ncoils,nframes);

    for frame = 1:nframes

        kdata(:,:,:,frame) = bsxfun(@times, ifft2( bsxfun(@times,b1,img(:,:,frame)) ) , mask(:,:,frame).*sqrt(n*m));

    end

end

function y = col(x)
y=x(:);
end