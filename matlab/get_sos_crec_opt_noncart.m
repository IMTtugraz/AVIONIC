function [u,crec] = get_sos_crec_opt_noncart(kdata, traj, dcf, imdims, gpu)

    if nargin < 5 
        gpu = 1;
        kernelWidth = 3.0;
        sectorWidth = 8.0;
        osf         = 2.0;
        FT  = gpuNUFFT([    real(traj(:)),imag(traj(:))]',...
                            ones(size(traj(:))),...
                            osf,kernelWidth,sectorWidth,imdims,[],true);
        col = @(x) x(:);
    else
    
        K           = floor(imdims*2);
        n_shift     = imdims/2;
        J           = [5,5];
        interptype  = 'kaiser';   %nufft interpolation kernel: 'minmax:kb', 'kaiser'

        om          = [real(traj(:)), imag(traj(:))];
        nufft_st_u0 = nufft_init(om*2*pi,imdims,J,K,n_shift,interptype);
    end
    
    [~,~,ncoils,~] = size(kdata);
    n = imdims(1);
    m = imdims(2);
    
    %Get coil-wise reconstruction	
    crec = zeros(n,m,ncoils);
    for j = 1:ncoils
        if gpu
            crec(:,:,j) =  FT'*(col( kdata(:,:,j,:)) .* dcf(:));
            
        else
            crec(:,:,j) =  nufft_adj(col(kdata(:,:,j,:)) .* dcf(:),nufft_st_u0)./ ...
                sqrt(prod(nufft_st_u0.Kd));
        end
    end
	
    %Get absolute value of u as sum of squares
    u = sqrt( sum( abs(crec).^2 , 3) );
	
    %Get phase of u by summation of crec
    u = abs(u).*exp(1i.*angle(sum(crec,3))); 	
