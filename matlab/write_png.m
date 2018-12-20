function  write_png( vid, outputdir, filenamebase, figurescale, nonorm)

if nargin < 5
    nonorm = 0;
end

vid = double(vid);

if nonorm
    imscale = figurescale;
else
    imscale = figurescale./max(abs(vid(:)));
end

eval(['mkdir ',outputdir]);

for i=1:size(vid,3)
    
    im = imscale.*abs(vid(:,:,i));
    im(im>1) = 1;
    
    imwrite(im,...
       [outputdir,'/',filenamebase,'_frame',num2str(i),'.png'],'png','compression','none');     
    
    
end

