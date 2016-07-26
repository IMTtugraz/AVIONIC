function [b,tvt,gap,par] = b1_grad(u0,crec,par_in)

%Set parameter############################################################

	
	%Set standard parameter
		
		%Ratio between primal and dual steps
		s_t_ratio = 1;
		
		sig = 1/sqrt(8);
		tau = 1/sqrt(8);

		%Stopping
		stop_rule = 'iteration';
		stop_par = 500;
		

				
		%Regularization parameters
		mu = 1;


		
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
		sig = sig*s_t_ratio;
		tau = tau/s_t_ratio;
	



%Algorithmic##################################################################
	
	tvt =0; gap = 0;
	
	%Image size
	[n,m,ncoils] = size(crec);
	

	%Primal variable
	b = zeros(n,m,ncoils);
		
	%Extragradient
	extb = b;
	
	%Dual variable
	p = zeros(n,m,2,ncoils);
	


	%Set up data##################################################################
	
		%Renormalize data range
		rg = max(abs(u0(:)));
		u0 = u0/rg;	
	
		%Transform given data to L^*( crec)
		Lw = zeros(n,m,ncoils);
		for j=1:ncoils

			Lw(:,:,j) = conj(u0) .* crec(:,:,j) ;
			
		end
		
		%Prepare 1 / (I + tau*abs(u0)^2) for proximity map
		I_u = zeros(n,m,ncoils);
		for j=1:ncoils
			I_u(:,:,j) = ones(n,m) ./ ( ones(n,m) + tau*abs(u0).^2 );
		end
	
	%{
	%TVT norm
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
	
	tvt(1) = b1_grad_primal(b,crec,u0,mu);
	%gap(1) = b1_gn_dual(p,q,r);
	%gap(1) = gap(1)./(n*m);
	enl = 1;
	%}
	
	
	k=0;
	go_on = 1;
while go_on
	
	%Algorithmic#########################
		

		%Dual ascent step
			p =  p + sig*fgrad_1( extb ) ;
			p = p/(1 + sig/mu);
			
				      	
		%Primal descent step

			%Descent
			extb = b - tau*(- bdiv_1(p));
		
			%Proximity map
			extb = ( extb + tau*Lw ) .* I_u;
	

		%Set extragradient
		b=2*extb - b;
    
    		%Swap extragradient and primal variable
		[b,extb] = deal(extb,b);
		
        	%Adapt stepsize
        	%[sig,tau] = steps_b1_grad(extb-b,sig,tau,s_t_ratio);
        	
        	
        	%Increment iteration number
        	k = k+1;
        	
        	%Check stopping rule
        	if ( strcmp(stop_rule,'iteration') && k>= stop_par )% || ( strcmp(stop_rule,'gap') && gap(1 + k/factor) < stop_par )
        		go_on = 0;
        	end
        	
        %Data collection##############################
        %{
        	if rem(k,10) == 0
        		display(['Iteration:    ',num2str(k)]);
        	end
        	
        	%Enlarge tvt and gap
        	if strcmp(stop_rule,'gap') && k>(1000*enl)
        		tvt = [tvt,zeros(1,1000)]; gap = [gap,zeros(1,1000)]; enl = enl + 1;
        	end
        	
        	if rem(k,factor) == 0
        		tvt(1 + k/factor) = b1_grad_primal(b,crec,u0,mu);
        		%gap(1 + k/factor) = b1_gn_dual(p,q,r);
        		%gap(1 + k/factor) = gap(1 + k/factor)./(n*m);
        	end
        %}	
        	
        	
        		
end
 
 %{      
       %Crop back
        if strcmp(stop_rule,'gap')
	        tvt = tvt(1:1+k);
        	gap = gap(1:1+k);
        end
        
   %}
   
        
        %Normalize b1 to one
        nb = sqrt(sum(abs(b).^2,3));
        for j=1:ncoils
        	b(:,:,j) = b(:,:,j)./nb;
        end
        
        display(['Sig:   ',num2str(sig)])
        display(['Tau:   ',num2str(tau)])
        display(['Nr-it: ', num2str(k)])
        





%Set results##########################################################################
	
	%Output: renormalized b

        
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


