clear; clc;

% Make sure you change file path

healthy_fileList = dir(fullfile("/Users/user/Documents/UMD2023/ENME485/Assignments/HW2/Reading Materials/Training/Healthy", "*.txt"));
faulty_fileList = dir(fullfile("/Users/user/Documents/UMD2023/ENME485/Assignments/HW2/Reading Materials/Training/Faulty", "*.txt"));
test_fileList = dir(fullfile("/Users/user/Documents/UMD2023/ENME485/Assignments/HW2/Reading Materials/Testing/", "*.txt"));

% Predefine a 38400 by 20 matrix for normal and faulty data
% 38400 per samples at 20 samples for each dataset
normal_data = zeros(38400, 20);
faulty_data = zeros(38400, 20);
test_data = zeros(38400, 30);

% Declare an empty array to store the first peak amplitude value after FFT
normal_amplitude = zeros(20,1);
faulty_amplitude = zeros(20,1);
test_amplitude = zeros(30,1);

% read from line 6 onwards to until the end of the txt file
datalines = [6, Inf];

% Loop through and read each file for healthy and faulty data
for i = 1:20
    healthy_fileName = "/Users/user/Documents/UMD2023/ENME485/Assignments/HW2/Reading Materials/Training/Healthy/" + healthy_fileList(i,1).name;
    bad_fileName = "/Users/user/Documents/UMD2023/ENME485/Assignments/HW2/Reading Materials/Training/Faulty/" + faulty_fileList(i,1).name;
    
    % Import the file data
    healthy_data = import_file_data(healthy_fileName, datalines);
    bad_data = import_file_data(bad_fileName, datalines);

    % Transcribe it to matrix
    for j = 1:38400
        normal_data(j,i) = healthy_data(j,1);
        faulty_data(j,i) = bad_data(j,1);
    end

end

% Loop through and read each file for test data
for i = 1:30
    test_fileName = "/Users/user/Documents/UMD2023/ENME485/Assignments/HW2/Reading Materials/Testing/" + test_fileList(i,1).name;
    
    % Import the file data
    temp_test_data = import_file_data(test_fileName, datalines);

    % Transcribe it to matrix
    for j = 1:38400
        test_data(j,i) = temp_test_data(j,1);
    end

end

%%

% Loop through each sample-set and perform FFT and record first peak
for i = 1:20
    Fs = 2560;            % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    L = 38400;            % Length of signal
    
    % Load healthy dataset values
    X = normal_data(:,i);

    f = Fs/L*(0:(L/2-1));
    
    % Perform FFT for Single-Sided Amplitude Spectrum
    Y = fft(X);
    P2 = abs(Y/L);
    P1 = P2(1:L/2);
    P1(2:end-1) = 2*P1(2:end-1);
        
    % Identify the peaks within 400 values
    pks = findpeaks(P1, 'MinPeakDistance',400);
    % Record the first peak
    normal_amplitude(i,1) = pks(1);
end

% Repeat but for faulty dataset
for i = 1:20
    Fs = 2560;            % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    L = 38400;            % Length of signal
    
    X = faulty_data(:,i);

    f = Fs/L*(0:(L/2-1));

    Y = fft(X);
    P2 = abs(Y/L);
    P1 = P2(1:L/2);
    P1(2:end-1) = 2*P1(2:end-1);
        
    pks = findpeaks(P1, 'MinPeakDistance',400);
    faulty_amplitude(i,1) = pks(1);
end

% Plot Feature View graph
plot(faulty_amplitude, '-ro')
grid on
title('Feature View');
xlabel('Sample Number');
ylabel('Amplitude');
hold on
plot(normal_amplitude, '-bo')
hold off
legend('Faulty Amplitude', 'Normal Amplitude');


%% Plot Testing Features

% Repeat but for test dataset
for i = 1:30
    Fs = 2560;            % Sampling frequency                    
    T = 1/Fs;             % Sampling period       
    L = 38400;            % Length of signal
    
    X = test_data(:,i);

    f = Fs/L*(0:(L/2-1));

    Y = fft(X);
    P2 = abs(Y/L);
    P1 = P2(1:L/2);
    P1(2:end-1) = 2*P1(2:end-1);
        
    pks = findpeaks(P1, 'MinPeakDistance',400);
    test_amplitude(i,1) = pks(1);
end

% Plot Feature View graph
plot(test_amplitude, '-ro')
grid on
title('Feature View');
xlabel('Sample Number');
ylabel('Amplitude');

%% DEBUGGING 
% Plot in time-domain

Fs = 2560;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 38400;            % Length of signal
t = (0:L-1)*T;        % Time vector

X = normal_data(:,1);

plot(t,X)
title("Data Acceleration")
xlabel("t (sec)")
ylabel("Acceleration")

%% DEBUGGING
% Plot Fast Fourier Transform

% Referred to https://www.mathworks.com/help/matlab/ref/fft.html

f = Fs/L*(0:(L/2-1));
Y = fft(X);
P2 = abs(Y/L);
P1 = P2(1:L/2);
P1(2:end-1) = 2*P1(2:end-1);

plot(f,P1) 
title("Single-Sided Amplitude Spectrum of fft(X)")
xlabel("f (Hz)")
ylabel("|P1(f)|")

% Find Peaks
[pks, locs] = findpeaks(P1, 'MinPeakDistance',400);
first_harmonic_peak = pks(1);

P1(locs(1))

%% Training

%% Select Training Portion and PCA (front/back)

%GoodSampleIndex (in this example I assume the baseline data is first 100
%samples

GoodSampleIndex=1:20;

%Lets assume the last 100 samples are from a degraded system
DegradedSampleIndex=21:40;

FeatureMatrix = [normal_amplitude; faulty_amplitude; test_amplitude];

%Baseline Data
BaselineData=FeatureMatrix(GoodSampleIndex,:);

DegradedData=FeatureMatrix(DegradedSampleIndex,:);



%% Train LR Model

%Label Vector (0.95 for good samples, 0.05 for bad samples
Label=[ones(size(BaselineData,1),1)*0.95; ones(size(DegradedData,1),1)*0.05];

%fit LR Model (glm-fit)
beta = glmfit([BaselineData; DegradedData],Label,'binomial');


%% Calculating Health Value (using LR Model)

%assume I have some test-data (assume it is samples 301-400)

TestSampleIndex = 41:70;

TestFeatureMatrix=FeatureMatrix(TestSampleIndex,:);

%calculate CV (Health Value)
CV_Test = glmval(beta,TestFeatureMatrix,'logit') ;  %Use LR Model
   
plot(CV_Test, '-bo')
grid on
title("CV of Testing Data Set")
xlabel("Testing Data Sample Number")
ylabel("CV")

