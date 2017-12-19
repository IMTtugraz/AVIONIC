function mri_obj = prepare_data_noncart(mri_obj, par_in, gpu)

    %Set parameter#############################################################
    % Standard parameters for artifial subsampling on self-generated datasets


    %Regularization parameters for coil construction
    u_reg = 1e-4;
    b1_reg = 2;
    b1_final_reg = 0.1;
    b1_final_nr_it = 1000;
    uH1mu = 1e-5;
    
    
    % non-uniform fft parameters
    J = [6,6];               % nufft kernelsize
    interptype = 'kaiser';   % nufft interpolation kernel: 'minmax:kb', 'kaiser'
    imgdims = [256,256];

    %Read parameter------------------------------------------------------------
    %Input: par_in-------------------------------------------------------------
    %Generate list of parameters
    vars = whos;
    for i=1:size(vars,1)
        par_list{i,1} = vars(i).name;
    end
    %Set parameters according to list
    for i=1:size(par_in,1);
        valid = false;
        for j=1:size(par_list,1); if strcmp(par_in{i,1},par_list{j,1})
                valid = true;
                eval([par_in{i,1},'=','par_in{i,2}',';']);
            end; end
        if valid == false; warning(['Unexpected parameter at ',num2str(i)]); end
    end
    %---------------------------------------------------------------------------
    
    if nargin <3
	gpu = 0;
    end
    [nsamplesonspoke,spokesperframe,ncoils,nframes] = size(mri_obj.data);
      
    mri_obj.data = mri_obj.data.*permute(repmat(sqrt(mri_obj.dcf),[1 1 1 ncoils]),[1 2 4 3]);

    N = imgdims;
    K = floor(N*1.5);
    n_shift = N/2;

    col = @(x) x(:);
    om = [real(mri_obj.traj(:)), imag(mri_obj.traj(:))];
    
    if gpu

       osf     = 1.5; % oversampling: 1.5 1.25
       wg      = 6; % kernel width: 5 7
       sw      = 8; % parallel sectors' width: 12 16
    
   	   FT = gpuNUFFT(om',(mri_obj.dcf(:)),osf,wg,sw,imgdims,[]);
       for j = 1 : ncoils
		  crec(:,:,j) = FT'*( col( mri_obj.data(:,:,j,:) ) )./nframes;
       end


    else
	    nufft_st_u0 = nufft_init(om*2*pi,N,J,K,n_shift,interptype);

	    % Calculate time-averaged image
	    % coil-wise time-averaged data from all spokes
	    % ATTENTION: consider sqrt(w) !
	    for j = 1 : ncoils
		crec(:,:,j) = nufft_adj( col( mri_obj.data(:,:,j,:) )...
		    		.* sqrt(mri_obj.dcf(:)) ,nufft_st_u0)./nframes./sqrt(prod(N));%nufft_st_u0.Kd));
	    end
	    clear nufft_st_u0;
    end
    [n,m,ncoils]        = size(crec);
    mri_obj.datadims    = [nsamplesonspoke,spokesperframe,ncoils,nframes];
    mri_obj.imgdims     = [n,m,ncoils];

    % Combine coil-wise time-averaged images for u0 (inital guess)
    if isfield(mri_obj,'b1')
        u0 = (sum(crec .* conj(mri_obj.b1), 3));
    else
        u0 = sqrt(sum(abs(crec).^2,3));
    end
    u0datanorm = abs(u0);
    datanorm = 255./median(u0datanorm(u0datanorm>=0.9.*max(u0datanorm(:))));
    
    mri_obj.data = mri_obj.data.*datanorm;
    mri_obj.datanorm = datanorm;
 
   
    % pre-calculate gridding kernels for time-frames --> input for forward_opt_noncart and backward_opt_noncart
    if ~isfield(mri_obj,'nufft_st') && ~gpu
      for frame=1:nframes
        om = [ real(col(mri_obj.traj(:,:,frame))), ...
               imag(col(mri_obj.traj(:,:,frame)))]*2*pi;
        nufft_st{frame} = nufft_init(om,N,J,K,n_shift,interptype);
      end
      mri_obj.nufft_st = nufft_st;
    end

       
    % coil sensitivity estimation
    if ~isfield(mri_obj,'b1')
        [mri_obj.b1,mri_obj.u0] = coil_construction_opt_noncart(mri_obj.data, mri_obj.traj, mri_obj.dcf,imgdims,...
        {'u_reg',u_reg;'b1_reg',b1_reg;'b1_final_reg',b1_final_reg;'b1_final_nr_it',b1_final_nr_it;'uH1mu', uH1mu});
        mri_obj.u0 = mri_obj.u0(:,:,ncoils);
    else
        mri_obj.u0 = u0.*datanorm;
    end
    
 
