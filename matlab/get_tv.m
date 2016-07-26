function [tv] = get_tv(x,ts,t1)
	
	%Get [ Du - v , E(v) ]
	x0 =  (1/ts)*fgrad_3_1( x(:,:,:,1),t1/ts );

    %Set norm
    tv = sum(sum(sum( norm_3( abs(x0(:,:,:,1:3)) ) ))) ;
end
