function [tvt] = get_ictgv2(x,alph0,alph1,alpha,ts,t1,ts2,t2)
	

	
	%Get [ Du - v , E(v) ]
	x0 =  cat( 4 ,  (1/ts)*fgrad_3_1( x(:,:,:,1) - x(:,:,:,5) , t1/ts ) - x(:,:,:,2:4)	,...
			(1/ts)*sym_bgrad_3_3( x(:,:,:,2:4) , t1/ts )				,...
			(1/ts2)*fgrad_3_1( x(:,:,:,5) , t2/ts2 ) - x(:,:,:,6:8)			,...
			(1/ts2)*sym_bgrad_3_3( x(:,:,:,6:8) , t2/ts2 )				);
			
	%Set norm
	tvt = (alpha/min(alpha,1-alpha))*(alph1*(sum(sum(sum( norm_3( abs(x0(:,:,:,1:3)) ) ))) )   + alph0*(sum(sum(sum( norm_6( abs(x0(:,:,:,4:9)) ) ))) ) )  +...
	      ((1-alpha)/min(alpha,1-alpha))*(alph1*(sum(sum(sum( norm_3( abs(x0(:,:,:,10:12)) ) ))) ) + alph0*(sum(sum(sum( norm_6( abs(x0(:,:,:,13:18)) ) ))) ));





end




