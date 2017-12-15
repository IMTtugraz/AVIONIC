function [sig,tau] = steps_ls_pd(x,sig,tau,K)

	% Get Kx
    Kx  = abs( fgrad_t(x(:,:,:,2)) );
    Kx2 = abs( K(x(:,:,:,1)+x(:,:,:,2)) );
    
    % Get |Kx|
    nKx = sqrt( sum( abs(Kx(:)).^2 ) + sum( abs(Kx2(:)).^2 ) );
    
    % Get |x|
    nx = sqrt(sum(sum(sum(	abs(x(:,:,:,1)).^2 + abs(x(:,:,:,2)).^2 )))); 
                  
    
    %Set |x| / |Kx|
    tmp = (nx/nKx);
    theta = 0.95;
    
    %Check convergence condition
    if sig*tau > tmp^2 %If stepsize is too large
        if theta^(2)*sig*tau < tmp^2 %Check if minimal decrease satisfies condition
            sig = theta*sig;
            tau = theta*tau;
        else                        %If not, decrease further
            sig = tmp;
            tau = tmp;
        end
    end
    
    
end


