function [g2,comp1,comp2,par,b1,u0,tvt,gap,g2_out,sig_out,tau_out,datanorm] = ictgv2_dmri_noncart(mri_obj,par_in)

    tic
    %Set path
    addpath(genpath('stand_functions'));

    %Set parameter############################################################

    %Set standard parameter
    %Ratio between primal and dual steps
    sig = 1/3;
    tau = 1/3;
    s_t_ratio = 1; %Was better than 0.2 in simple test,ld=200,t1=0.4

    %(Slower) Debug mode
    debug = 0;

    %Time/space weigth ratios
    time_space_weight1 = 4;        % tested
    time_space_weight2 = 0.5;      % tested

    %Balancing term in (0,1): with b=min(alpha,1-alph), ICTV = (alpha/b)*TGV1 + ((1-alpha)/b)*TGV2
    alpha =  0.5;   % tested

    unequal_steps = 1; %allow different (not-switched) stepsizes for the second gradient

    %Stopping
    stop_rule = 'iteration';
    stop_par = 500;

    %TGV parameter
    alph0 = sqrt(2);
    alph1 = 1;

    %Regularization parameter
    ld = 4; %Optimal for subfac = 5
    adapt_ld = 1; %Adapt parameter according to given data
    steig = 0.33;
    dist = 6;

    %Regularization parameters for coil construction
    u_reg = 1e-4;
    b1_reg = 2;
    b1_final_reg = 0.2; % changed for non-cartesian 
    b1_final_nr_it = 1000;
    uH1mu =  1e-5;
	

    %Non-uniform FFT parameter
    J           = [5 5];       % nufft kernelsize
    interptype  = 'kaiser';    % nufft interpolation kernel: 'minmax:kb', 'kaiser'
    imgdims     = [256,256];

    eltime = 0;

    %Read parameter-------------------------------------------------------------------------
    %Input: par_in--------------------------------------------------------------------------
    %Generate list of parameters
    vars = whos;
    for l=1:size(vars,1)
        par_list{l,1} = vars(l).name;
    end
    %Set parameters according to list
    for l=1:size(par_in,1);
        valid = false;
        for j=1:size(par_list,1); if strcmp(par_in{l,1},par_list{j,1})
                valid = true;
                eval([par_in{l,1},'=','par_in{l,2}',';']);
            end; end
        if valid == false; warning(['Unexpected parameter at ',num2str(l)]); end
    end
    %---------------------------------------------------------------------------------------
    %---------------------------------------------------------------------------------------


    %Update parameter dependencies
    %Set Gradient parameters
    [w1,w2] = get_weights_3d(time_space_weight1);
    ts = 1/w1;
    t1 = 1/w2;

    [w1,w2] = get_weights_3d(time_space_weight2);
    ts2 = 1/w1;
    t2 = 1/w2;

    %Stepsize
    sig = sig*s_t_ratio;
    tau = tau/s_t_ratio;

    if unequal_steps == false
        ts2 = t1;
        t2 = ts;
    end

    %Initialize###########################################

    %Set zero output
    comp1=0;comp2=0;b1=0;tvt=0;gap=0;g2_out=cell(1);sig_out=0;tau_out=0;datanorm=0;

    [nsamplesonspoke,spokesperframe,ncoils,nframes] = size(mri_obj.data);
    n = imgdims(1); m = imgdims(2);
    
    % Prepare rawdata
    [mri_obj] = prepare_data_noncart(mri_obj,{...                                      ...
                                      'u_reg',u_reg; ...
                                      'b1_reg',b1_reg; ...
                                      'b1_final_reg', b1_final_reg; ...
                                      'b1_final_nr_it',b1_final_nr_it ; ...
                                      'uH1mu',  uH1mu;	...
                                      'J',J;...                     
                                      'interptype',interptype;...
                                      'imgdims',imgdims});
     u0 = mri_obj.u0;
     b1 = mri_obj.b1;


    % check adjointness
%     K = @(x) backward_opt_noncart(x, mri_obj.nufft_st, mri_obj.b1, mri_obj.dcf);
%     Kh = @(x) forward_opt_noncart( x, mri_obj.nufft_st, mri_obj.b1, mri_obj.dcf);
%     xx = randn(n,m,nframes);
%     yy = randn(nsamplesonspoke,spokesperframe,ncoils,nframes);
%     Kx = K(xx);
%     Khy = Kh(yy);
%     adjval = dot(yy(:),Kx(:)) - dot(Khy(:),xx(:));
%     
    %Adapt regularization parameter ( needs maybe other heuristics for non-cart)
    if adapt_ld == 1
        subfac = n/spokesperframe/pi*2;
        ld = subfac*steig + dist;
        display(['Adapted ld to: ',num2str(ld)]);
    end

    %Algorithmic##################################################################

    %Primal variable
    x = zeros(n,m,nframes,8);

    for j=1:nframes
        x(:,:,j,1) = mri_obj.u0;
    end

    %Extragradient
    ext = x;

    %Dual variable
    y = zeros(n,m,nframes,18);	%Ordered as: (1) (2) (3)   (1,1) (2,2) (3,3) (1,2) (1,3) (2,3)
    z = zeros(nsamplesonspoke,spokesperframe,ncoils,nframes);

    %Finite difference operants
    [fDx,fDy,fDz] = get_sp_fdif(n,m,nframes);
    [bDx,bDy,bDz] = get_sp_bdif(n,m,nframes);


    %########################################################################################################
    if debug%Only in debug mode------------------------------------------------------------------------------
        factor = 1;
        if strcmp(stop_rule,'iteration');
            factor = 5;
            tvt = zeros(1, floor( stop_par/factor ) );
            gap = zeros(1, floor( stop_par/factor ) );
        elseif strcmp(stop_rule,'gap');
            tvt = zeros(1,1000);
            gap = zeros(1,1000);
        else
            error('Wrong stopping rule');
        end
        tvt(1) = get_ictgv2(x,alph0,alph1,alpha,ts,t1,ts2,t2);
        gap(1) = abs( tvt(1) + gstar_ictgv2_dmri_noncart(x,y,z,mri_obj,ts,t1,ts2,t2,ld) );

        gap(1) = gap(1)./(n*m*nframes);
        enl = 1;
    end%------------------------------------------------------------------------------------------------------
    %########################################################################################################

    k=0;
    go_on = 1;
    while go_on

        %Algorithmic#########################


        %Dual ascent step (tested)
        y = y + sig*cat(4,	sp_grad_3_1( ext(:,:,:,1) - ext(:,:,:,5), fDx,fDy,fDz,ts,t1 ) - ...
            ext(:,:,:,2:4)	,...
            sp_sym_grad_3_3( ext(:,:,:,2:4), bDx,bDy,bDz,ts,t1 )				,...
            sp_grad_3_1( ext(:,:,:,5), fDx,fDy,fDz,ts2,t2 ) - ext(:,:,:,6:8)	,...
            sp_sym_grad_3_3( ext(:,:,:,6:8), bDx,bDy,bDz,ts2,t2 )				);


        z = z + sig*( backward_opt_noncart( ext(:,:,:,1), mri_obj.nufft_st, mri_obj.b1, mri_obj.dcf ));

        %Proximity maps
        n1 = norm_3( abs(y(:,:,:,1:3)) );
        n1 = max(1,n1./alph1);
        for i=1:3
            y(:,:,:,i) = y(:,:,:,i)./n1;
        end
        n1 = norm_6( abs(y(:,:,:,4:9)) );
        n1 = max(1,n1./alph0);
        for i=4:9
            y(:,:,:,i) = y(:,:,:,i)./n1;
        end
        n1 = norm_3( abs(y(:,:,:,10:12)) );
        n1 = max(1,n1./alph1);
        for i=10:12
            y(:,:,:,i) = y(:,:,:,i)./n1;
        end
        n1 = norm_6( abs(y(:,:,:,13:18)) );
        n1 = max(1,n1./alph0);
        for i=13:18
            y(:,:,:,i) = y(:,:,:,i)./n1;
        end

        z = (z-sig*mri_obj.data ) / (1+sig/ld);

        %Primal descent step (tested)
        ext = x - tau*cat(4, ...
            -sp_div_3_3( y(:,:,:,1:3),fDx,fDy,fDz,ts,t1 ) + forward_opt_noncart			(z,mri_obj.nufft_st,mri_obj.b1,mri_obj.dcf)		,...
            -y(:,:,:,1:3) - sp_div_3_6( y(:,:,:,4:9),bDx,bDy,bDz,ts,t1 )					,...
            sp_div_3_3( y(:,:,:,1:3),fDx,fDy,fDz,ts,t1 ) - sp_div_3_3( y(:,:,:,10:12),fDx,fDy,fDz,ts2,t2 ) 	,...
            -y(:,:,:,10:12) - sp_div_3_6( y(:,:,:,13:18),bDx,bDy,bDz,ts2,t2 )				);



        %Set extragradient
        x=2*ext - x;

        %Swap extragradient and primal variable
        [x,ext] = deal(ext,x);

        %Adapt stepsize
        if (k<10) || (rem(k,50) == 0)
            [sig,tau] = steps_ictgv2_dmri_noncart(ext-x,sig,tau,ts,t1,ts2,t2,s_t_ratio,mri_obj);
        end


        %Increment iteration number
        k = k+1;

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
                tvt(1 + k/factor) = get_ictgv2(x,alph0,alph1,alpha,ts,t1,ts2,t2);
                gap(1 + k/factor) = abs( tvt(1 + k/factor) + gstar_ictgv2_dmri_noncart...
                    (x,y,z,mri_obj,ts,t1,ts2,t2,ld) );
                gap(1 + k/factor) = gap(1 + k/factor)./(n*m*nframes);
            end
        end%------------------------------------------------------------------------------------------------------
        %########################################################################################################


    end

    display(['Sig:   ',num2str(sig)])
    display(['Tau:   ',num2str(tau)])
    display(['Nr-it: ', num2str(k)])


    g2       =  sig*x(:,:,:,1);
    comp2    =  sig*x(:,:,:,5);
    comp1    =  g2 - comp2;
    par      =  par_in;
    datanorm =  mri_obj.datanorm;
    
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



end











