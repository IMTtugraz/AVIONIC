function [ reco ] = bart_grasp(data, b1, spf, ld, traj, scale, maxspokes )

if size(data,2) > maxspokes
    data= data(:,1:maxspokes,:);
end

nframes = floor(size(data,2)/spf);

data = data(:,1:nframes*spf,:);
if ndims(data)<4
    data = permute(data,[1 2 4 3]);
end
if ndims(b1)<4
    b1 = permute(b1,[1 2 4 3]);
end

[nread,nspokes,~,ncoils] = size(data);

if ~isempty(traj)
    if ~isreal(traj)
        traj_(1,:,:) = real(traj);
        traj_(2,:,:) = imag(traj);
        traj = traj_; clear traj_;
    end
    traj(3,:,:) = zeros(size(traj(1,:,:)));
end

% normalize traj to bart standard [-256 255];
traj = traj-min(traj(:));
traj = traj*511./max(traj(:));
traj = traj - 256;

writecfl('graspbart',data);
writecfl('graspbartsens',b1);
writecfl('tmp1',traj(:,:,1:nframes*spf));

outfile = ['out',num2str(floor(now*1e4))];
unix(['./bart_grasp.sh --outfile=',outfile,' --spf=',num2str(spf),' --phases=',num2str(nframes),' --lambda=',num2str(ld),' --scale=',num2str(scale)]);

reco = squeeze(readcfl(outfile));

unix(['rm ',outfile,'*'])

end

