function w = goldcmp(k,type)

[N,nspokes] = size(k);

switch type
    
    case 'ramlak'
        w = ones(N,1);
        w = abs(linspace(-N/2,N/2,N))';
        w = w.*pi/4/nspokes;
        w = repmat(w,[1 nspokes]);
        
    case 'pipe'
        Nd = [size(k,1) size(k,1)]/2;
        Jd = [6,6];
        Kd = floor([Nd*1.5]);
        n_shift = Nd/2;

        omrev_all = [col(real(k)), col(imag(k))]*2*pi;
        strev_all = nufft_init(omrev_all, Nd, Jd, Kd, n_shift,'kaiser');
 
        AH_all = @(x) nufft_adj(x,strev_all)./sqrt(prod(Nd));
        A_all = @(x) nufft(x,strev_all)./sqrt(prod(Nd));
        
        w = ones(N*nspokes,1);
        niter = 5;   
        for i = 1:niter
            goal = A_all(AH_all(w));
            w = w./abs(goal);
        end
        
        w = reshape(w, [N, nspokes]);
end
     
end