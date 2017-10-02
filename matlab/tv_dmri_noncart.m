function [g2,par,comps,b1,tvt,gap,sos] = tv_dmri_noncart(dataname,par_in)

    %Set path
    addpath(genpath('./stand_functions'))	
    tic
    
    %Set parameter############################################################
    %Set standard parameter
		
    %Ratio between primal and dual steps
    sig = 1/3;
    tau = 1/3;
    s_t_ratio = 1;

    %Time/space weigth ratios
    time_space_weight = 6.5;  % tested

    %Stopping
    stop_rule = 'iteration';
    stop_par = 500;

    %Regularization parameter
    ld = 4*0.2991;
    adapt_ld = 1;

    steig = 0.33;   % tested, ssimroi
    dist = 6;         % tested, ssimroi

    %Use debug mode
    write_cuda = 0;
    debug = 0;

    %Code type for b1 field reconstruction
    b1_code = 'const';
    b1_source = 0;
    u0_source = 0;

    %Regularization parameters for coil construction
    u_reg = 1E-4;
    b1_reg = 2;  
    b1_final_reg = 1;
    b1_final_nr_it = 1000;	
    uH1mu = 1E-5;
    
    %Parameter for read_mri
    crop = 0;                                          % crop data to accelerate calculations
    crop_factor = 31;                              % crop factor  
    acc = 0;                                            % acceleration factor
    refL = 0;                                            % reference lines or block in k-space center
    sampling_pattern = 'vista';  	% sampling scheme {'cart','randomcart','random_cart_xy'}
                                        % cart: interleaved cartesian
                                        % randomcart: phase-encoding randomized cartesian
                                        % randomcart_xy: phase- and frequency-encoding randomized cartesian	 

    coil_cmp = 0;                                   % coil compression via SVD to reduce computation time 
    coil_threshold = 0.05;                      % still in testing phase (introduces inconsitencies in virtual coils)
    coil_cmp_method = 'dacl';               % that cause artifacts in reconstruction

    save_refsos = 0;                                                 % write reference and sum-of-squares reconstruction before and after artifical
    save_refsos_path = './script_results/refsos';       % undersampling on full-datasets
    b1_code_refsos = 'walsh';                                   % estimate coil-sensitivities from full data for reference 
    phase_sensitive_refsos = 0;
    normalize_data = 1;
    datanorm = 1;
    
    spokesperframe = 21;
    J = [5 5];               %nufft kernelsize
    interptype = 'kaiser';   %nufft interpolation kernel: 'minmax:kb', 'kaiser'

    perfusion = 0;

    eltime = 0;
    init_time = 0;

    %Set zero output
    comps=cell(1);b1=0;tvt=0;gap=0;g2_out=cell(1);sig_out=0;tau_out=0;

    %Read parameter-------------------------------------------------------------------------
    %Input: par_in--------------------------------------------------------------------------
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
    %---------------------------------------------------------------------------------------
    %---------------------------------------------------------------------------------------			

    %Update parameter dependencies

    %Stepsize
    sig = 1/3;
    tau = 1/3;
    sig = sig*s_t_ratio;
    tau = tau/s_t_ratio;

    [w1,w2] = get_weights_3d(time_space_weight); 

    ts = 1/w1;
    t1 = 1/w2;

    %The values:
    %time_space_weight = 5;ld = 4*0.2991;steig = 0.4*0.2991;dist = 10*0.2991;
    %have been tested and are equivalent to setting:
    %ts = 1;t1 = 0.2;ld=4;steig=0.4;dist=10;

    %Standard reconstruction###########################################

    %Set zero output
    b1=0;tvt=0;gap=0;

    %Read data
    [mri_obj,sos,b1_full,datanorm] = read_mri_noncart(dataname,{'crop',crop;...
                                      'crop_factor',crop_factor;...
                                      'spokesperframe',spokesperframe;...
                                      'J',J;...                     
                                      'interptype',interptype;...
                                      'debug',debug; ...
                                      'write_cuda',write_cuda});
    
    display('Using coil_construction code for b1 field...');
    [mri_obj.b1,mri_obj.u0] = coil_construction_opt_noncart(mri_obj.data, ...
        mri_obj.k,mri_obj.w,[mri_obj.imgdims(1) mri_obj.imgdims(2)],...
        {'uH1mu',uH1mu;'u_reg',u_reg;'b1_reg',b1_reg;'b1_final_reg',b1_final_reg;'b1_final_nr_it',b1_final_nr_it});

    display('Finished coil calculation...');
    
    nsamplesonspoke = mri_obj.datadims(1);
    spokesperframe = mri_obj.datadims(2);
    ncoils = mri_obj.datadims(3);
    nframes = mri_obj.datadims(4);
    n = mri_obj.imgdims(1);
    m = mri_obj.imgdims(2);

    init_time = toc
    tic

    %Adapt regularization parameter ( needs maybe other heuristics for non-cart)
    if adapt_ld == 1
        subfac = n/spokesperframe/pi*2;
        ld = subfac*steig + dist;
        display(['Adapted ld to: ',num2str(ld)]);
    end

    %Algorithmic##################################################################

    %Primal variable
    x = zeros(n,m,nframes,'single');
    for j=1:nframes
        x(:,:,j,1) = mri_obj.u0(:,:,ncoils);
    end
    clear u0;

    %Extragradient
    ext = x;

    %Dual variable
    y = zeros(n,m,nframes,3,'single');
    z = zeros(nsamplesonspoke,spokesperframe,ncoils,nframes,'single');

    %Finite difference operants
    [fDx,fDy,fDz] = get_sp_fdif(n,m,nframes);
    [bDx,bDy,bDz] = get_sp_bdif(n,m,nframes);

    %########################################################################################################
    if debug%Only in debug mode------------------------------------------------------------------------------
        %TVT norm
        factor = 1;
        if strcmp(stop_rule,'iteration');
            factor = 10;
            tvt = zeros(1, floor( stop_par/factor ) );
            gap = zeros(1, floor( stop_par/factor ) );
        elseif strcmp(stop_rule,'gap');
            tvt = zeros(1,1000);
            gap = zeros(1,1000);
        else
            error('Wrong stopping rule');
        end
        tvt(1) = get_tv(x,ts,t1);
        gap(1) =  abs( tvt(1) + gstar_tv_mri_motion_noncart(x,y,z,mri_obj,ts,t1,ld) );
        gap(1) = gap(1)./(n*m*nframes);
        enl = 1;
    end%------------------------------------------------------------------------------------------------------
    %########################################################################################################

    k=0;
    go_on = 1;
    while go_on
        %Algorithmic#########################

        %Dual ascent step
        y = y + sig*sp_grad_3_1( ext, fDx,fDy,fDz,ts,t1 );

        z = z + sig*( backward_opt_noncart( ext, mri_obj.nufft_st, mri_obj.b1, mri_obj.w, spokesperframe ));

        %Proximity maps
        n1 = norm_3( abs(y(:,:,:,1:3)) );
        n1 = max(1,n1);
        for i=1:3
            y(:,:,:,i) = y(:,:,:,i)./n1;
        end
        z = (z-sig*mri_obj.data ) / (1+sig/ld);

        %Primal descent step
        ext = x - tau*( -sp_div_3_3( y,fDx,fDy,fDz,ts,t1 ) + forward_opt_noncart(z,mri_obj.nufft_st,mri_obj.b1,mri_obj.w) );

        %Set extragradient
        x=2*ext - x;

            %Swap extragradient and primal variable
        [x,ext] = deal(ext,x);

        %Adapt stepsize
        if (k<10) || (rem(k,50) == 0)
            [sig,tau] = steps_tv_mri_motion_noncart(ext-x,sig,tau,ts,t1,s_t_ratio,mri_obj);
        end

        %Increment iteration number
        k = k+1;

        %Display iteration
        if rem(k,10) == 0
            display(['Iteration:    ',num2str(k)]);
        end

        %Check stopping rule
        if ( strcmp(stop_rule,'iteration') && k>= stop_par )% || ( strcmp(stop_rule,'gap') && gap(1 + k/factor) < stop_par )
            go_on = 0;
        end

        %########################################################################################################	
        if debug%Only in debug mode------------------------------------------------------------------------------
            %Enlarge tvt and gap
            if strcmp(stop_rule,'gap') && k>(1000*enl)
                tvt = [tvt,zeros(1,1000)]; gap = [gap,zeros(1,1000)]; enl = enl + 1;
            end

            if rem(k,factor) == 0
                tvt(1 + k/factor) = get_tv(x,ts,t1);
                gap(1 + k/factor) = abs( tvt(1 + k/factor) + gstar_tv_mri_motion_noncart(x,y,z,mri_obj,ts,t1,ld) );%
                gap(1 + k/factor) = gap(1 + k/factor)./(n*m*nframes);
            end
        end%------------------------------------------------------------------------------------------------------
        %########################################################################################################	
    end	

    display(['Sig:   ',num2str(sig)])
    display(['Tau:   ',num2str(tau)])
    display(['Nr-it: ', num2str(k)])

    g2 =  x(:,:,:,1);

    %########################################################################################################
    if debug%Only in debug mode------------------------------------------------------------------------------
        %Crop back
        if strcmp(stop_rule,'gap')
            tvt = tvt(1:1+k);
            gap = gap(1:1+k);
        end
        %Output of b1 field
        b1 = mri_obj.b1;
    end%------------------------------------------------------------------------------------------------------
    %########################################################################################################

    eltime = toc;

    %Write parameter-------------------------------
    %Input: k (iteration number)-------------------
    psz = size(par_list,1);
    for i=1:psz
        par{i,1} = par_list{i,1};
        eval(['par{i,2} = ',par_list{i,1},';'])
    end
    par{psz+1,1} = 'iteration_nr'; par{psz+1,2}=k;
    par{psz+2,1} = mfilename;
    %Output: par-----------------------------------
    %----------------------------------------------
end

