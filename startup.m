addpath('lib');

cwd = pwd();
try
 update_mex('lib/randsample_stratified_cpp.cpp');
catch err
 display(err)
end
cd(cwd)


