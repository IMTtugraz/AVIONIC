function norm = getNorm(data_c, data_l, mask)
%function for data normalization
  [x,y,z,~,~] = size(data_c);
  N = size(data_c,ndims(data_c));
  
  mask_ = squeeze(mask(:,:,:,1,:));
  for coil = 1:size(data_c,4)
    data_ = sum(squeeze(data_c(:,:,:,coil,:)),4);
    msum = sum(mask_,4);
    data_(msum > 0) = data_(msum>0)./msum(msum>0);
    crec(:,:,:,coil) = fftn(data_)/sqrt(x*y*z);   
  end
  u = sqrt(sum(abs(crec).^2, 4));
  datanorm_c = 1000./median(u(u>=0.95.*max(u(:))));
  for coil = 1:size(data_l,4)
    data_ = sum(squeeze(data_l(:,:,:,coil,:)),4);
    msum = sum(mask_,4);
    data_(msum > 0) = data_(msum>0)./msum(msum>0);
    crec(:,:,:,coil) = fftn(data_)/sqrt(x*y*z);
  end
  u = sqrt(sum(abs(crec).^2, 4));
  datanorm_l = 1000./median(u(u>=0.95.*max(u(:))));
  norm = 1./min(datanorm_c,datanorm_l);
end