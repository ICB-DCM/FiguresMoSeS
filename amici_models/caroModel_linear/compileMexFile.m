% This compile script was automatically created from Python SBML import.
% If mex compiler is set up within MATLAB, it can be run from MATLAB 
% in order to compile a mex-file from the Python generated C++ files.

modelName = 'caroModel_linear';
amimodel.compileAndLinkModel(modelName, '', [], [], [], []);
amimodel.generateMatlabWrapper(2, 1, 5, 1, 0, 0, ...
    [], ['simulate_' modelName '.m'], modelName, ...
    'lin', 1, 1);