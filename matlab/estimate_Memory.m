function enough_memory = estimate_Memory(data)

[x,y,z,nc,t] = size(data);

cf_size = 8; %size of complex float on GPU 

raw_data_size = x*y*z*nc*t*cf_size;
coil_size = x*y*z*nc*cf_size;
mask_size = x*y*z*t*cf_size;

img_size = x*y*z*t*cf_size;

N = x*y*z*t*cf_size;
	  
all = raw_data_size * 2 + coil_size + mask_size + img_size * 2 + ...
      18 * N + ... %temp variables
	  16 * 4 * N + ... %Grad variables
	  10 * 4 * N + ... %SymGrad variables
	  4 * raw_data_size;
  
  enough_memory = all/(1024*1024);
	  
% 
% 	  
% 	  imgTemp %N
% 	  div1Temp %N
% 	  div3Temp %N
% 	  div2Temp % 4*N
% 	  y1Temp % 4*N
% 	  y3Temp % 4*N
% 	  y7Temp % 4*N
% 	  y2Temp % 10*N
% 	  y5Temp % data_gpu size
% 	  tempSum % N	  
% 	  CVector ext1; %N
%       CVector x1_old; %N
% 
%   std::vector<CVector> x2; %4*N
%   std::vector<CVector> ext2; %4*N
%   std::vector<CVector> x2_old; %4*N
%   CVector ext3; %N
%   CVector x3_old; %N
%   std::vector<CVector> x4; %4*N
%   std::vector<CVector> ext4; %4*N
%   std::vector<CVector> x4_old; %4*N
%   std::vector<CVector> x5; %4*N
%   std::vector<CVector> ext5; %4*N
%   std::vector<CVector> x5_old; %4*N
%   
%     std::vector<CVector> y1; %4*N
%   std::vector<CVector> y2; % 10*N
%   std::vector<CVector> y3; %4*N
%   std::vector<CVector> y4; % 10*N
%   CVector y5; %coils*N
%   CVector y6; %coils*N
%   std::vector<CVector> y7; %4*N
%   std::vector<CVector> y8; % 10*N
% 	  
% 	  
% 	  CVector norm_gpu; %N
%   CVector temp; %N
 
end