function [sig,tau] = steps_ictgv2_dmri(x,sig,tau,ts,t1,ts2,t2,s_t_ratio,mri_obj)

	[n,m] = size(x);

	Kx = zeros(n,m,6);    
	
	%Get Kx
    	Kx = abs( cat(4,		(1/ts)*fgrad_3_1( x(:,:,:,1) - x(:,:,:,5) , t1/ts ) - x(:,:,:,2:4)	,...
					(1/ts)*sym_bgrad_3_3( x(:,:,:,2:4) , t1/ts )				,...
				  	(1/ts2)*fgrad_3_1( x(:,:,:,5) , t2/ts2 ) - x(:,:,:,6:8)			,...
				  	(1/ts2)*sym_bgrad_3_3( x(:,:,:,6:8) , t2/ts2 )				) );
				  	
	Kx2 = abs( backward_opt( x(:,:,:,1),mri_obj.mask,mri_obj.b1) );
    
    	%Get |Kx|
    	nKx = sqrt(	sum(sum(sum(	Kx(:,:,:,1).^2 + Kx(:,:,:,2).^2 + Kx(:,:,:,3).^2 	+...
    					Kx(:,:,:,4).^2 + Kx(:,:,:,5).^2 + Kx(:,:,:,6).^2 	+...
    					2*Kx(:,:,:,7).^2 + 2*Kx(:,:,:,8).^2 + 2*Kx(:,:,:,9).^2  +...
    					Kx(:,:,:,10).^2 + Kx(:,:,:,11).^2 + Kx(:,:,:,12).^2 	+...
    					Kx(:,:,:,13).^2 + Kx(:,:,:,14).^2 + Kx(:,:,:,15).^2 	+...
    					2*Kx(:,:,:,16).^2 + 2*Kx(:,:,:,17).^2 + 2*Kx(:,:,:,18).^2  ))) 	+...
    			sum(		Kx2(:).^2						)	);
    	
    	%Get |x|
    	nx = sqrt(sum( abs(x(:)).^2 ));
    	                    
    
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
