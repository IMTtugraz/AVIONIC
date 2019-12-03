function [sig,tau] = steps_tv_mri_motion_noncart(x,sig,tau,ts,t1,s_t_ratio,mri_obj)
%Function to get the adaptive stepsize for tgv3_mjpeg function

    [n,m] = size(x);

    %Get Kx
    Kx = abs(	(1/ts)*fgrad_3_1( x , t1/ts )	);
    Kx2 = abs( backward_opt_noncart( x,mri_obj.nufft_st,mri_obj.b1,mri_obj.w,mri_obj.datadims(2) ) );
    %Get |Kx|
    nKx = sqrt(	sum( Kx(:).^2 ) + sum( Kx2(:).^2 ) );

    %Get |x|
    nx = sqrt(sum( abs(x(:)).^2 ) );


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

    %sig = tmp;
    %tau = tmp;

end
