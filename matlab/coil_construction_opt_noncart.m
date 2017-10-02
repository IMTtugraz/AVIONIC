function [b1,u,ind,par] = coil_construction_opt_noncart(kdata,traj,dcf,imdims,par_in)

    u_reg = 0.2;
    b1_reg = 2;
    b1_final_reg = 0.1; % changed for non-cart
    b1_final_nr_it = 1000;
    uH1mu =  1e-1;

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


    %Initialization of mri data
    [nsamplesonspoke,spokesperframe,ncoils,nframes] = size(kdata);
    n = imdims(1);
    m = imdims(2);

    %Initialize data
    b1 = zeros(n,m,ncoils);
    ind = zeros(ncoils,1);
    u = zeros(n,m,ncoils);


    %Get standard reconstructions
    [u0,crec] = get_sos_crec_opt_noncart(kdata,traj,dcf,imdims);

    %Get absolute value of sensitivities
    absb1 = abs( b1_from_u_h1_direct( abs(u0),abs(crec) ,uH1mu ) );


    %Get initial index
    [mx,pos] = max( squeeze(sum(sum(abs(crec),1),2)) );
    ind(1) = pos;
    display(['Index: ',num2str(ind(1))]);



    %Initialize first b1 field with zero phase
    b1(:,:,ind(1)) = absb1(:,:,ind(1));

    %Get u to first b1 field
    u(:,:,1) = u_tgv2_opt(b1(:,:,ind(1)),crec(:,:,ind(1)),zeros(n,m),{'stop_par',100;'nu',u_reg});%0.1!!!!!


    dcf = zeros(n,m);
    for j=2:ncoils

        %Update weights
        dcf = sqrt( dcf.^2 + absb1(:,:,ind(j-1)).^2 );

        %Get index of next b1 field
        inters = zeros(ncoils,1);
        for k = 1:ncoils
            if ~ismember(k,ind)
                inters(k) = sum(sum( abs(crec(:,:,ind(j-1))).*abs(crec(:,:,k)) ));
            end
        end
        [mx,ind(j)] = max(inters);
        display(['Index: ',num2str(ind(j))]);


        %Get new b1
        phs = b1_grad_opt(dcf.*absb1(:,:,ind(j)).*u(:,:,j-1),dcf.*crec(:,:,ind(j)),{'stop_par',500;'mu',b1_reg});
        b1(:,:,ind(j)) = absb1(:,:,ind(j)).*exp(i.*angle(phs)); %Abs

        %Get new image
        u(:,:,j) = u_tgv2_opt( b1(:,:,ind(1:j)) , crec(:,:,ind(1:j)),u(:,:,j-1) ,{'stop_par',100;'nu',u_reg});




    end

    %B1 correction
    b1 = b1_grad_opt(u(:,:,ncoils),crec,{'stop_par',b1_final_nr_it;'mu',b1_final_reg});



    %Output: sensitivities b1, image u


    %Write parameter-------------------------------
    psz = size(par_list,1);
    for l=1:psz
        par{l,1} = par_list{l,1};
        eval(['par{l,2} = ',par_list{l,1},';'])
    end
    par{psz+1,1} = mfilename;
    %Output: par-----------------------------------
    %----------------------------------------------









