function griddingrecon = make_gridding_recon(kdata,traj,b1,nspokesperframe,gpu)

[nsamplesonspoke,nspokes,ncoils,nframes] = size(kdata);

if nargin < 5
    gpu = 0;
end

if ndims(kdata)==4
    
    nframes = size(kdata,4);
    kdatau = kdata;
    traju = traj;
    for frame=1:nframes
        densu(:,:,frame) = goldcmp(traju(:,:,frame),'ramlak');
    end
    
elseif ndims(kdata)<4
    
    % number of frames
    nframes = floor(nspokes/nspokesperframe);
    
    kdata = kdata(:,1:nframes*nspokesperframe,:);
    
    % trajectory
    if ndims(traj) == 3
        traj = traj(1,:,:)+1i.*traj(2,:,:);
        traj = squeeze(traj);
    end
    traj = traj(:,1:nframes*nspokesperframe);
    
    
    for frame=1:nframes
        kdatau(:,:,:,frame) = kdata(:,(frame-1)*nspokesperframe+1:frame*nspokesperframe,:);
        traju(:,:,frame)    = traj(:,(frame-1)*nspokesperframe+1:frame*nspokesperframe);
        densu(:,:,frame)    = goldcmp(traju(:,:,frame),'ramlak');
    end
    
    
end
shift   = [0,0];
fftSize = [nsamplesonspoke/2,nsamplesonspoke/2];


for frame=1:nframes
    if gpu
        osf     = 1.5; % oversampling: 1.5 1.25
        wg      = 6; % kernel width: 5 7
        sw      = 8; % parallel sectors' width: 12 16
        
        FT_crec = gpuNUFFT([    real(col(traju(:,:,frame))), ...
            imag(col(traju(:,:,frame)))]',...
            ones(nsamplesonspoke*nspokesperframe,1),osf,wg,sw,fftSize,[]);
        
    else
        FT_crec = NUFFT(traju(:,:,frame), 1, 1, shift, fftSize, 2);
    end
    
    clear recon_crec;
    recon_crec = zeros(fftSize(1),fftSize(2),ncoils);
    
    for coil=1:ncoils
        if gpu
            
            recon_ =  ...
                FT_crec'*col(densu(:,:,frame).*double(kdatau(:,:,coil,frame)));
        else
            recon_ =  ...
                FT_crec'*(densu(:,:,frame).*double(kdatau(:,:,coil,frame)));
        end
        fprintf('frame: %d/%d Ncoil: %d/%d \n',frame,nframes,coil,ncoils)
        recon_crec(:,:,coil) = recon_;
    end
    
    if ~isempty(b1)
        griddingrecon(:,:,frame) = sum(conj(b1).*recon_crec,3);
    else
        griddingrecon(:,:,frame) = sqrt(sum(abs(recon_crec).^2,3));
    end
end


end
