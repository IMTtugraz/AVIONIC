function [g2,par,b1,tvt,gap,g2_out,sig_out,tau_out] = tgv2_dmri(data, par_in)
%TGV2_DMRI:
% Spatio-temporal Total Generalized Variation
% for accelerataed dynamic MRI reconstruction
%
%   input:
%    - data: dynamic mri data (center k-space)
%            (dimensions: (phase encoding, frecquency encoding, coils, frames)
%    - par_in: reconstruction parameter: set {} to use standard parameters as below
%
%   output:
%    - g2:      TGV reconstruction 
%    - par:     Updated input parameter
%
%    (debug)
%    - b1:      Estimated coil-sensitivities
%    - tvt:     
%    - gap:     Estimated primal-dual Gap
%    - g2_out:  ICTGV reconstruction every 50 iterations
%    - sig_out: Primal step-size every iteration
%    - tau_out: Dual step-size every iteration
%
% Reference:
% [1] Matthias Schloegl, Martin Holler,
%     Andreas Schwarzl, Kristian Bredies and Rudolf Stollberger.
%     "Infimal Convolution of Total Generalized Varitaion Functionals
%     for dynamic MRI".
%     Magnetic Resonance in Medicine, in press, 2016. 
%     http://imsc.uni-graz.at/mobis/publications/SFB-Report-2016-002.pdf
%
% [2] Matthias Schloegl, Martin Holler, Kristian Bredies and Rudolf Stollberger.
%     "A Variational Approach for Coil-Sensitivity Estimation for Undersampled 
%      Phase-Sensitive Dynamic MRI Reconstruction"
%      Proc. Intl. Soc. Mag. Reson. Med. 23, Toronto, Canada
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
    tic
    %Set parameter#########################################################

    %Ratio between primal and dual steps
    s_t_ratio = 1;

    %Time/space weigth ratios
    time_space_weight = 6;  % tested

    %(Slower) Debug mode
    debug = 1;

    %Stopping
    stop_rule = 'iteration';
    stop_par = 500;

    %TGV parameter
    alph0 = sqrt(2);
    alph1 = 1;

    %Regularization parameter
    ld = 5.804591;     % Adapted automatically
    adapt_ld = 1;      % Adapt parameter according to given data
    steig = 0.223253;  % tested
    dist = 5.581338;   % tested


    %Regularization parameters for coil construction
    u_reg = 0.2;
    b1_reg = 2;
    b1_final_reg = 0.1;
    b1_final_nr_it = 1000;
    uH1mu = 1e-5;
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


    %Stepsize
    sig = 1/3;
    tau = 1/3;
    sig = sig*s_t_ratio;
    tau = tau/s_t_ratio;

    [w1,w2] = get_weights_3d(time_space_weight);

    ts = 1/w1;
    t1 = 1/w2;


    %Initialize###########################################

    %Set zero output
    b1=0;tvt=0;gap=0;g2_out=cell(1);sig_out=0;tau_out=0;

 
    %Setup Data and estimate sensitivities
    mri_obj = prepare_data(data, {'u_reg',u_reg; 'b1_reg', b1_reg; 'b1_final_reg', b1_final_reg; ...
        'b1_final_nr_it',b1_final_nr_it; 'uH1mu', uH1mu;});
    clear data;


    %Get size
    [n,m,ncoils,nframes] = size(mri_obj.data);

    %Adapt regularization parameter
    if adapt_ld == 1
        subfac = (n*m*ncoils)/sum(sum(sum(mri_obj.mask,3)));
        ld = subfac*steig + dist;
        display(['Adapted ld(acceleration) to: ',num2str(ld)]);
    end


    %Algorithmic##################################################################

    %Primal variable
    x = zeros(n,m,nframes,4);
    for j=1:nframes
        x(:,:,j,1) = mri_obj.u0;
    end


    %Extragradient
    ext = x;

    %Dual variable
    y = zeros(n,m,nframes,9);	%Ordered as: (1) (2) (3)   (1,1) (2,2) (3,3) (1,2) (1,3) (2,3)
    z = zeros(n,m,ncoils,nframes);

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
        tvt(1) = get_tgv2(x,alph0,alph1,ts,t1);
        gap(1) = abs( tvt(1) + gstar_tgv2_dmri(x,y,z,mri_obj,ts,t1,ld) );
        gap(1) = gap(1)./(n*m*nframes);
        enl = 1;

        sig_out = zeros(stop_par,1);
        tau_out = zeros(stop_par,1);

    end%------------------------------------------------------------------------------------------------------
    %########################################################################################################


    k=0;
    go_on = 1;
    while go_on

        %Dual ascent step (tested)
        y = y + sig*cat(4,	sp_grad_3_1( ext(:,:,:,1), fDx,fDy,fDz,ts,t1 ) - ext(:,:,:,2:4)	,...
            sp_sym_grad_3_3( ext(:,:,:,2:4), bDx,bDy,bDz,ts,t1 )				);


        z = z + sig*( backward_opt( ext(:,:,:,1),mri_obj.mask,mri_obj.b1 ) );


        %Proximity maps
        n1 = norm_3( abs(y(:,:,:,1:3)) );
        n1 = max(1,n1./alph1);
        for i=1:3
            y(:,:,:,i) = y(:,:,:,i)./n1;
        end
        n2 = norm_6( abs(y(:,:,:,4:9)) );
        n2 = max(1,n2./alph0);
        for i=4:9
            y(:,:,:,i) = y(:,:,:,i)./n2;
        end

        z = (z-sig*mri_obj.data ) / (1+sig/ld);

        %Primal descent step (tested)
        ext = x - tau*cat(4, -sp_div_3_3( y(:,:,:,1:3),fDx,fDy,fDz,ts,t1 ) + forward_opt(z,mri_obj.mask,mri_obj.b1)		,...
            -y(:,:,:,1:3) - sp_div_3_6( y(:,:,:,4:9),bDx,bDy,bDz,ts,t1 )	);



        %Set extragradient
        x=2*ext - x;

        %Swap extragradient and primal variable
        [x,ext] = deal(ext,x);

        %Adapt stepsize
        if (k<10) || (rem(k,50) == 0) || debug
            [sig,tau] = steps_tgv2_dmri(ext-x,sig,tau,ts,t1,s_t_ratio,mri_obj);
            sig_out(k+1) = sig;
            tau_out(k+1) = tau;
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
                tvt(1 + k/factor) = get_tgv2(x,alph0,alph1,ts,t1);
                gap(1 + k/factor) = abs( tvt(1 + k/factor) + ...
                            gstar_tgv2_dmri(x,y,z,mri_obj,ts,t1,ld) );
                gap(1 + k/factor) = gap(1 + k/factor)./(n*m*nframes);
            end

            if rem(k,50) == 0
                g2_out{k/50} = x(:,:,:,1);
            end
        end%------------------------------------------------------------------------------------------------------
        %########################################################################################################


    end

    display(['Sig:   ',num2str(sig)])
    display(['Tau:   ',num2str(tau)])
    display(['Nr-it: ', num2str(k)])


    g2 =  x(:,:,:,1);
    comp1 = ext(

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
    for l=1:psz
        par{l,1} = par_list{l,1};
        eval(['par{l,2} = ',par_list{l,1},';'])
    end
    par{psz+1,1} = 'iteration_nr'; par{psz+1,2}=k;
    par{psz+2,1} = mfilename;
    %Output: par-----------------------------------
    %----------------------------------------------


end













