function [ parset_best,error_all,par_all,gap_all,ictgvnorm_all,datafid_all,datanorm_all ] = test_ictgv_modelparameter( mri_obj, reference, roi, alpharange, tsw1range, tsw2range, par_in )


% alpharange = linspace(0.1,0.9,5);
% tsw1range = linspace(1,10,5);
% tsw2range = linspace(0,3,5);
% roi = ones(size(reference));

roi = logical(roi);
if numel(roi) ~= numel(reference)
   roi = repmat(roi,[1 1 size(reference,3)]); 
end

mkdir ./modtest
dt = datestr(now,'yymmdd_HHMM');
savedir = ['./modtest/test',dt];
eval(['mkdir ',savedir]);

kk=1;
for i=1:length(alpharange)
    for j=1:length(tsw1range)
        for k=1:length(tsw2range)
    
            if ~(exist([savedir,'/parset',num2str(kk)])==2)
            par_in_mod = {'alpha',alpharange(i);... 
                     'timeSpaceWeight',tsw1range(j);...
                     'timeSpaceWeight2',tsw2range(k)};

            par_inin = [par_in;par_in_mod];

            [ g2, comp1, ~, ~, u0, pdgap, datanorm, ictgvnorm, datafid ] = ...     
                avionic_matlab_gpu(mri_obj,par_inin);

            g2 = permute(g2,[2 1 3])./datanorm;
            
            [errorv, errorvroi] = myerror(abs(g2),abs(reference),'nrmse',roi);
            [ssimv,ssim_map] = ssim(abs(g2),abs(reference));
   			ssimvroi = mean(ssim_map(roi));

            save([savedir,'/parset',num2str(kk)],...
                'datanorm','par_inin','datanorm','comp1','g2','comp1','pdgap','ictgvnorm','datafid','par_in','errorv','errorvroi','ssimv','ssimvroi');
         
            error_all(kk,1) = errorv;
            error_all(kk,2) = errorvroi;
            error_all(kk,3) = 1-ssimv;
            error_all(kk,4) = 1-ssimvroi;
            
            par_all{kk}.par_in = par_inin;
            gap_all{kk}.gap = pdgap;
            ictgvnorm_all{kk}.ictgvnorm = ictgvnorm;    
            datafid_all{kk}.datafid = datafid;
            
            datanorm_all(kk) = datanorm;
            
            end
       
            kk=kk+1;
        end
    end
end

[~,idx] = max(mean(error_all,2));

parset_best = par_all{idx};



end

