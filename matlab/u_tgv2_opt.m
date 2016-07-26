function [u,tvt,gap,par] = u_tgv2_opt(b1,crec,u0,par_in)
    %Set parameter############################################################


    %Set standard parameter

    %Ratio between primal and dual steps
    s_t_ratio = 1;


    %Stopping
    stop_rule = 'iteration';
    stop_par = 500;

    %TGV parameter
    alph0 = sqrt(2);
    alph1 = 1;


    %Regularization parameters
    nu = 1;



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
    sig = sqrt(1/12);
    tau = sqrt(1/12);
    sig = sig*s_t_ratio;
    tau = tau/s_t_ratio;




    %Algorithmic##################################################################

    tvt = 0; gap =0;

    %Image size
    [n,m,ncoils] = size(crec);


    %Primal variable
    u = u0;
    v = zeros(n,m,2);

    %Extragradient
    extu = u;
    extv = v;

    %Dual variable
    p = zeros(n,m,2);
    q = zeros(n,m,3);



    %Set up data##################################################################



    %Transform given data to L^*( crec)
    Lw = sum( conj(b1).*crec, 3);

    %Prepare 1 / (I + tau*sum(abs(b1)^2)) for proximity map
    I_b = ones(n,m) ./ ( ones(n,m) + tau*sum( abs(b1).^2,3) );




    k=0;
    go_on = 1;
    while go_on

        %Algorithmic#########################


        %Dual ascent step
        p = p + sig*( fgrad_1( extu ) - extv );
        q = q + sig*( sym_bgrad_2( extv ) );

        %Projections
        np = sqrt( sum (abs(p).^2,3) );
        np = max(1,np./(alph1*nu));
        p(:,:,1) = p(:,:,1)./np; p(:,:,2) = p(:,:,2)./np;

        nq = sqrt( abs(q(:,:,1)).^2 + abs(q(:,:,2)).^2 + 2*abs(q(:,:,3)).^2 );
        nq = max(1,nq./(alph0*nu));
        q(:,:,1) = q(:,:,1)./nq; q(:,:,2) = q(:,:,2)./nq;  q(:,:,3) = q(:,:,3)./nq;


        %Primal descent step

        %Descent
        extu = u - tau*(- bdiv_1(p));
        extv = v - tau*( - p - fdiv_2(q));

        %Proximity map
        extu = ( extu + tau*Lw ) .* I_b;


        %Set extragradient
        u=2*extu - u;
        v=2*extv - v;

        %Swap extragradient and primal variable
        [u,extu] = deal(extu,u);
        [v,extv] = deal(extv,v);


        %Increment iteration number
        k = k+1;

        %Check stopping rule
        if ( strcmp(stop_rule,'iteration') && k>= stop_par )% || ( strcmp(stop_rule,'gap') && gap(1 + k/factor) < stop_par )
            go_on = 0;
        end

        %Data collection##############################





    end


    display(['Sig:   ',num2str(sig)])
    display(['Tau:   ',num2str(tau)])
    display(['Nr-it: ', num2str(k)])






    %Set results##########################################################################

    %Output: reconstructed timeconstant u


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


