function [norm3] = norm_3(v)

	norm3 = sqrt( v(:,:,:,1).^2 + v(:,:,:,2).^2 + v(:,:,:,3).^2 );

end
