function K = read_ismrmrd(filename)
% adapted from https://github.com/ismrmrd/ismrmrd/blob/master/examples/matlab/test_recon_dataset.m
% dependency: ismrmrmd https://github.com/ismrmrd/

    if exist(filename, 'file')
        dset = ismrmrd.Dataset(filename, 'dataset');
    else
        error(['File ' filename ' does not exist. Please generate it.'])
    end
    hdr = ismrmrd.xml.deserialize(dset.readxml);

    %% Encoding and reconstruction information
    try
        nSlices = hdr.encoding.encodingLimits.slice.maximum + 1;
    catch
        nSlices = 1;
    end
    try
        nCoils = hdr.acquisitionSystemInformation.receiverChannels;
    catch
        nCoils = 1;
    end
    try
        nReps = hdr.encoding.encodingLimits.repetition.maximum + 1;
    catch
        nReps = 1;
    end
    try
        nContrasts = hdr.encoding.encodingLimits.contrast.maximum + 1 + 1;
    catch
        nContrasts = 1;
    end
    %% Read all the data

    D = dset.readAcquisition();
    % ignore noise scans
    isNoise = D.head.flagIsSet('ACQ_IS_NOISE_MEASUREMENT');
    firstScan = find(isNoise==0,1,'first');
    if firstScan > 1
        noise = D.select(1:firstScan-1);
    else
        noise = [];
    end
    meas = D.select(firstScan:D.getNumber);
    clear D;

    % read data
    for rep = 1:nReps
        for contrast = 1:nContrasts
            for slice = 1:nSlices
                % Initialize the K-space storage array
                K = zeros(enc_Nx, enc_Ny, enc_Nz, nCoils);
                % Select the appropriate measurements from the data
                acqs = find( (meas.head.idx.contrast==(contrast-1)) ...
                    & (meas.head.idx.repetition==(rep-1)) ...
                    & (meas.head.idx.slice==(slice-1)));
                for p = 1:length(acqs)
                    ky = meas.head.idx.kspace_encode_step_1(acqs(p)) + 1;
                    kz = meas.head.idx.kspace_encode_step_2(acqs(p)) + 1;
                    K(:,ky,kz,:) = meas.data{acqs(p)};
                end

            end
        end
    end
