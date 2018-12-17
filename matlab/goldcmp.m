function dcf = goldcmp(k,type)

[N,nspokes] = size(k);

switch type
    
    case 'ramlak'
        dcf = ones(N,1);
        dcf = abs(linspace(-N/2,N/2,N))';
        dcf = dcf.*pi/4/nspokes;
        dcf = repmat(dcf,[1 nspokes]);
        
    case 'pipe'
        Nd = [size(k,1) size(k,1)]/2;
        Jd = [6,6];
        Kd = floor([Nd*1.5]);
        n_shift = Nd/2;
        
        omrev_all = [col(real(k)), col(imag(k))]*2*pi;
        strev_all = nufft_init(omrev_all, Nd, Jd, Kd, n_shift,'kaiser');
        
        AH_all = @(x) nufft_adj(x,strev_all)./sqrt(prod(Nd));
        A_all = @(x) nufft(x,strev_all)./sqrt(prod(Nd));
        
        dcf = ones(N*nspokes,1);
        niter = 5;
        for i = 1:niter
            goal = A_all(AH_all(dcf));
            dcf = dcf./abs(goal);
        end
        dcf = reshape(dcf, [N, nspokes]);
        
    case 'pipegpu'
        imwidth = size(k,1);%/2;
        osf     = 2; % oversampling: 1.5 1.25
        wg      = 3; % kernel width: 5 7
        sw      = 8; % parallel sectors' width: 12 16
        
        FT_dyn  = gpuNUFFT([col(real(k)),col(imag(k))]',ones(size(col(k))),...
            osf,wg,sw,[imwidth,imwidth],[],true);
        
        dcf = ones(N*nspokes,1);
        niter = 5;
        for i = 1:niter
            goal = FT_dyn*(FT_dyn'*(dcf));
            dcf = dcf./abs(goal);
        end
        dcf = reshape(dcf, [N, nspokes]);
        
    case 'voronoi'
        fac = N*pi/2/nspokes;
        dcf = DoCalcDCF(real(k(:)), imag(k(:)));
        dcf = fac*dcf';
        dcf = reshape(dcf, [N, nspokes]);
        
    case 'china'
        assert(ndims(k)>1 & ndims(k)<3, 'k must be 2D'); %#ok<*ISMAT>
        [nro, nvw] = size(k);
        dk = abs(k(end,1)-k(1,1))/(nro-1); % spacing between k points
        
        % angular compensation
        angcmp = goldangcomp(nvw);
        dcf = repmat(angcmp', [nro, 1]);
        dcf(k==0) = 1/4/nvw;
        disp(sum(sum(dcf)));
        
        % radial compensation
        dcf = dcf.*abs(k)/dk;
        dcf(k==0) = 1/4/nvw;
end

end

%% HELPER FUNCTIONS
function DCF=DoCalcDCF(Kx, Ky)
    % caluclate density compensation factor using Voronoi diagram

    % remove duplicated K space points (likely [0,0]) before Voronoi
    K = Kx + 1i*Ky;
    [K1,m1,n1]=unique(K);
    K = K(sort(m1));

    % calculate Voronoi diagram
    [K2,m2,n2]=unique(K);
    Kx = real(K2);
    Ky = imag(K2);
    Area = voronoiarea(Kx,Ky);

    % use area as density estimate
    DCF = Area(n1);

    % take equal fractional area for repeated locations (likely [0,0])
    % n   = n1;
    % while ~isempty(n)
    %     rep = length(find(n==n(1)));
    %     if rep > 1
    %         DCF (n1==n(1)) = DCF(n1==n(1))./rep;
    %     end
    %     n(n==n(1))=[];
    % end

    % normalize DCF
    DCF = DCF ./ max(DCF);

    % figure; voronoi(Kx,Ky);


end

function Area = voronoiarea(Kx,Ky)
    % caculate area for each K space point as density estimate

    Kxy = [Kx,Ky];
    % returns vertices and cells of voronoi diagram
    [V,C] = voronoin(Kxy);

    % compute area of each ploygon
    Area = zeros(1,length(Kx));
    for j = 1:length(Kx)
        x = V(C{j},1); 
        y = V(C{j},2);
        % remove vertices outside K space limit including infinity vertices from voronoin
        x1 = x;
        y1 = y;
        ind=find((x1.^2 + y1.^2)>0.25);
        x(ind)=[]; 
        y(ind)=[];
        % calculate area
        lxy = length(x);
        if lxy > 2
            ind=[2:lxy 1];
            A = abs(sum(0.5*(x(ind)-x(:)).*(y(ind)+y(:))));
        else
            A = 0;
        end
        Area(j) = A;
    end
end

function angcmp = goldangcomp(nvw)
% Compute density compensation for Golden Angle radial trajectories
% input:
%   nsp    - Number of spokes
%
% output:
%   angcmp - [nx1], angular areas occupied by each radial spoke
%
% Yulin V Chang, PENN, 20150113
%

% First, check if n is a Fibonacci number
load fibonacci
goldrat = (sqrt(5)-1)/2;
flen = length(fibonacci);
if (nvw>fibonacci(flen))
    print('number too big, regenerate Fibonacci series');
    exit(1);
elseif nvw<=5
    print('number too small. Go figure this out yourself!');
    exit(1);
end

ii = flen;
while nvw<fibonacci(ii) % if nsp=fibonacci(flen) then ii=flen
    ii = ii-1;
end % now nsp >= fibonacci(ii)
m = nvw - fibonacci(ii);
comp = zeros(nvw,1);
if m==0
    comp(                1:fibonacci(ii-2)) = 1+goldrat;
    comp(fibonacci(ii-2)+1:fibonacci(ii-1)) = 2;
    comp(fibonacci(ii-1)+1:            end) = 1+goldrat;
else % 0 < m < fibonacci(ii-1)
    comp(      1:  m) = 1;
    comp(nvw-m+1:nvw) = 1;
    if m<=fibonacci(ii-3)
        comp(fibonacci(ii-2)+1:fibonacci(ii-1)) = 2; % initialization
        comp(m+1:fibonacci(ii-2)+m) = 1+goldrat;
        comp(fibonacci(ii-1)+1:fibonacci(ii)) = 1+goldrat;
    else
        comp(                m+1:fibonacci(ii-1)  ) = 1+goldrat;
        comp(fibonacci(ii-1)  +1:fibonacci(ii-2)+m) = 2*goldrat;
        comp(fibonacci(ii-2)+m+1:fibonacci(ii)    ) = 1+goldrat;
    end
end

angcmp = comp/sum(comp); % convert to absolute areas/pi


end