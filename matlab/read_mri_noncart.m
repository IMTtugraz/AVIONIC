function [mri_obj,sos,b1,datanorm] = read_mri_noncart(filename,par_in)

    %Set parameter#############################################################
    addpath(genpath('/windows/C/Users/Schloegl/Documents/1Projekte/MRI_motion/data'));

    % Standard parameters for artifial subsampling on self-generated datasets
    crop = 0;                              % crop data to accelerate calculations
    crop_factor = 4;                    % crop factor

    coil_cmp = 0;                       % apply coil-compression (not tested)
    coil_threshold = 0.05;          % defines threshold for coil reduction
    coil_cmp_method = 'dacl';   % defines coil compression method

    %Regularization parameters for coil construction
    u_reg = 0.2;
    b1_reg = 2;
    b1_final_reg = 0.1; % changed for non-cart
    b1_final_nr_it = 1000;
    uH1mu =  1e-1;
	

    % non-uniform fft parameters
    spokesperframe = 21;
    J = [5 5];                      %nufft kernelsize
    interptype = 'kaiser';   %nufft interpolation kernel: 'minmax:kb', 'kaiser'

    write_cuda = 0;
    debug = 0;

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
        if valid == false; warning(['Unexpected parameter at ',num2str(i),': ',par_in{i,1}]); end
    end


    % Read dataset#############################################################
    % sort data according to chosen number of spokes per frame

    eval(['load ',filename]);
    
    if (crop)
        kdata = kdata(floor(end/4):floor(3*end/4),1:end/crop_factor,1:floor(end/crop_factor));
        w = w(floor(end/4):floor(3*end/4),1:floor(end/crop_factor));
        k = k(floor(end/4):floor(3*end/4),1:floor(end/crop_factor));
        b1 = b1(:,:,floor(end/crop_factor))
    end
    
    nframes = floor(size(kdata,2)/spokesperframe);
    ncoils = size(kdata,3);
    
    % sort golden angle frames
    k_ = []; w_=[];
    for ii=1:nframes
        data(:,:,:,ii)= kdata(:,(ii-1)*spokesperframe+1:ii*spokesperframe,:);
        k_(:,:,ii)=  k(:,(ii-1)*spokesperframe+1:ii*spokesperframe);
        w_(:,:,ii)= w(:,(ii-1)*spokesperframe+1:ii*spokesperframe);
    end
    k = k_; clear k_; w = w_; clear w_

    % coil sensitivity estimation
    [mri_obj.b1,u0] = coil_construction_opt_noncart(data,k,w,[size(b1,1) size(b1,2)],...
        {'u_reg',u_reg;'b1_reg',b1_reg;'b1_final_reg',b1_final_reg;'b1_final_nr_it',b1_final_nr_it;'uH1mu', uH1mu});

    %Calculate time-averaged image
    N = [size(mri_obj.b1,1) size(mri_obj.b1,2)];
    K = floor(N*2);
    n_shift = N/2;

    col = @(x) x(:);
    om = [real(k(:)), imag(k(:))];
    nufft_st_u0 = nufft_init(om*2*pi,N,J,K,n_shift,interptype);

    % coil-wise time-averaged data from all spokes
    % ATTENTION: consider sqrt(w) !
    for j =1:ncoils
        g1(:,:,j) = nufft_adj( col( data(:,:,j,:) ) .* col(w) ,nufft_st_u0)./sqrt(prod(nufft_st_u0.Kd));
    end
    clear nufft_st_u0;

    [n,m,ncoils]  = size(g1);
    nsamplesonspoke = size(data,1);
    mri_obj.datadims = [nsamplesonspoke,spokesperframe,ncoils,nframes];
    mri_obj.imgdims = [n,m,ncoils];

    % Combine coil-wise time-averaged images for u0 (inital guess)
    u0 = (sum(g1 .* conj(mri_obj.b1), 3));
    u0datanorm = abs(u0);
    datanorm = 255./median(u0datanorm(u0datanorm>=0.9.*max(u0datanorm(:))));
    clear u crec

    mri_obj.data = data.*datanorm.*permute(repmat(sqrt(w),[1 1 1 ncoils]),[1 2 4 3]);
    mri_obj.k = k;
    mri_obj.w = w;
    mri_obj.u0 = u0;
    clear data k w u0

    % pre-calculate gridding kernels for time-frames --> input for forward_opt_noncart and backward_opt_noncart
    for i=1:nframes
        om = [real(col(mri_obj.k(:,(i-1)*spokesperframe+1:i*spokesperframe))), ...
            imag(col(mri_obj.k(:,(i-1)*spokesperframe+1:i*spokesperframe)))]*2*pi;
        nufft_st{i} = nufft_init(om,N,J,K,n_shift,interptype);
    end
    mri_obj.nufft_st = nufft_st;
    clear nufft_st

    % test forward and backward operation
    if (debug)

        coilnorm = sqrt(sum(abs(mri_obj.b1).^2,3));
        mri_obj.b1 = mri_obj.b1./repmat(coilnorm,[1 1 ncoils]);

        % cartesian
        K_cart = @(x) backward_opt( x, ones(n,n,nframes), mri_obj.b1);
        Kh_cart = @(x) forward_opt( x, ones(n,n,nframes), mri_obj.b1);

        [ adj_val,sumcheckimg,opnorm ] = check_adj(K_cart,Kh_cart,[n,n,nframes],[n,n,ncoils,nframes])

        scale = 1;%200;
        % non-cartesian
        K = @(x) scale.*backward_opt_noncart(x, mri_obj.nufft_st, mri_obj.b1, mri_obj.w, spokesperframe);
        Kh = @(x) forward_opt_noncart( scale.*x, mri_obj.nufft_st, mri_obj.b1, mri_obj.w);

        [ adj_val,sumcheckimg,opnorm ] = check_adj(K,Kh,[n,n,nframes],size(mri_obj.data))

    end

    if (write_cuda)
        % permute in order to compensate matlab column major order

        % b1
        writebin_vector(permute(mri_obj.b1,[1 2 3]),...
            ['./CUDA/data/breathhold_perf_b1_' sprintf('%d_%d_%d',n,m,ncoils)  '.bin']);

        % data
        writebin_vector(reshape(mri_obj.data,[nspokes*nsamplesonspoke ncoils nframes]),...
            ['./CUDA/data/breathhold_perf_data_' sprintf('%d_%d_%d_%d',nsamplesonspoke,nspokes,ncoils,nframes) '.bin']);

        % trajectory
        % split traj in x1 x2 x3 .... y1 y2 y3 order (needed by gpuNUFFT!)
        kk = reshape([real(col(mri_obj.k)), imag(col(mri_obj.k))],[nspokes*nsamplesonspoke*nframes 2]);
        kk = reshape(kk,[nspokes*nsamplesonspoke nframes 2]);
        writebin_vector(permute(kk,[1 3 2]),...
            ['./CUDA/data/breathhold_perf_k_' sprintf('%d_%d_%d',nsamplesonspoke,nspokes,nframes) '.bin']);

        % density compensation
        writebin_vector(col(mri_obj.w),...
            ['./CUDA/data/breathhold_perf_w_' sprintf('%d_%d_%d',nsamplesonspoke,nspokes,nframes) '.bin']);

        % initial guess    
        writebin_vector(mri_obj.u0,...
            ['./CUDA/data/breathhold_perf_u0_' sprintf('%d_%d',n,m) '.bin']);

    end

    % make sos reconstruction
    sos = forward_opt_noncart( mri_obj.data, mri_obj.nufft_st, mri_obj.b1, mri_obj.w);

end