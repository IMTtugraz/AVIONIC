function [sig,tau] = steps_tgv2_dmri_noncart(x,sig,tau,ts,t1,s_t_ratio,mri_obj)

   [n,m] = size(x);

    Kx = zeros(n,m,6);

    %Get Kx
    Kx = abs( cat( 4 ,  	(1/ts)*fgrad_3_1( x(:,:,:,1) , t1/ts ) - x(:,:,:,2:4)	,...
        (1/ts)*sym_bgrad_3_3( x(:,:,:,2:4) , t1/ts		 	) ) );

    Kx2 = abs( backward_opt_noncart( x(:,:,:,1),mri_obj.nufft_st,mri_obj.b1,mri_obj.w,mri_obj.datadims(2) ) );
    
    %Get |Kx|
    nKx = sqrt(	sum(sum(sum( 	Kx(:,:,:,1).^2 + Kx(:,:,:,2).^2 + Kx(:,:,:,3).^2 + ...
        Kx(:,:,:,4).^2 + Kx(:,:,:,5).^2 + Kx(:,:,:,6).^2 + ...
        2*Kx(:,:,:,7).^2 + 2*Kx(:,:,:,8).^2 + 2*Kx(:,:,:,9).^2  ))) + ...
        sum(		Kx2(:).^2 )   							);

    %Get |x|
    nx = sqrt(sum(sum(sum(	abs(x(:,:,:,1)).^2 + abs(x(:,:,:,2)).^2 + abs(x(:,:,:,3)).^2 + abs(x(:,:,:,4)).^2  ))));


    %Set |x| / |Kx|
    tmp = (nx/nKx);
    theta = 0.95;

    %Check convergence condition
    if sig*tau > tmp^2 %If stepsize is too large
        if theta^(2)*sig*tau < tmp^2 %Check if minimal decrease satisfies condition
            sig = theta*sig;
            tau = theta*tau;
        else                        %If not, decrease further
            sig = tmp*s_t_ratio;
            tau = tmp/s_t_ratio;
        end
    end


end
