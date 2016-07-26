function [ v ] = readbin_vector( filename, datatype)
%READBIN_VECTOR reads a (column) vector from a binary file
%   If filename is not set the file can be choosen with an open dialog

    v = [];
    path = '';

    if nargin == 0
        [file, path] = uigetfile({'*.bin','Binary vector file (*.bin)';'*.*',  'All Files (*.*)'},'Load vector ...');
        filename = fullfile(path, file);
    end
    
    if isequal(filename,0) || isequal(path,0)
        % user pressed cancel
        return;
    end
    
    % default type is set to double
    if nargin < 2 
        datatype = 'double';
    end
    
    % try to open a file for reading (binary)
    [fid, message] = fopen(filename, 'r');
    if fid == -1
        error(message);
    end
    
    % read complex header info
    iscomplex = fread(fid, 1, 'uchar');
    
    % read number of vector elements (size)
    size = fread(fid, 1, 'uint32');
    
    % read vector data
    v = fread(fid, size, datatype);
    if bitand(iscomplex, 1) > 0
        tmp_imag_data = fread(fid, size, datatype);
        v = complex(v, tmp_imag_data);
    end
    
    fclose(fid);
    
end
