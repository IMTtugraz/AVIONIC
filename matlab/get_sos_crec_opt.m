function [u,crec] = get_sos_crec_opt(kdata,mask)

    [n,m,ncoils,nframes] = size(kdata);
  
    msum = sum(mask,3);

     for j = 1:ncoils
 
         data = sum( squeeze(kdata(:,:,j,:)), 3);
 
         data(msum>0) = data(msum>0)./msum(msum>0);
 
         crec(:,:,j) = fft2( data )./sqrt(n*m);
         
     end
    
    %Get absolute value of u as sum of squares
    u = sqrt( sum( abs(crec).^2 , 3) );

    %Get phase of u by summation of crec
    u = abs(u).*exp(1i.*angle(sum(crec,3)));
    
