function [tgv] = get_tgv2(x,alph0,alph1,ts,t1)

	%Get [ Du - v , E(v) ]
	x0 =  cat( 4 ,  (1/ts)*fgrad_3_1( x(:,:,:,1),t1/ts ) - x(:,:,:,2:4), (1/ts)*sym_bgrad_3_3( x(:,:,:,2:4), t1/ts ) );
	%Set norm
    tgv = alph1*(sum(sum(sum( norm_3( abs(x0(:,:,:,1:3)) ) ))) ) + alph0*(sum(sum(sum( norm_6( abs(x0(:,:,:,4:9)) ) ))) );

end
