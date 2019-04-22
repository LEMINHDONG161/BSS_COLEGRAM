    %*************** setup sources **********
format compact

%**** if you are mixing the sources yourself:
cd('C:\Users\jys\Desktop\BSS and Antenna\Passive sonar\Program_final\BSS Fuction\BasicICA')
sources=readsounds(['word2';'word1']); % see "help readsounds"
sources=readsounds(['word2';'word1';'whdru';'whis1';'whis2';'wittg';'whdr2';'whdr3']); % see "help readsounds"
  % write your own code here, since readsounds looks for audiofiles.
  % All you want is a NxP matrix (N=no of mixtures/sources, P=no. of data points)
  
  
rng(42);

% Knobs
n   = 1000;             % # samples
T   = [3, 4, 5];        % # periods for each signal
SNR = 50;               % Signal SNR
d   = 3;                % # mixed observations
r   = 3;                % # independent/principal components

% Generate ground truth
t        = @(n,T) linspace(0,1,n) * 2 * pi * T;
Ztrue(1,:) = sin(t(n,T(1)));            % Sinusoid
Ztrue(2,:) = sign(sin(t(n,T(2))));      % Square

sources = Ztrue;
  
[N,P]=size(sources);                 % P=17408, N=2, for example
permute=randperm(P);                 % generate a permutation vector
s=sources(:,permute);                % time-scrambled inputs for stationarity

a=[1 2; 1 1]                         % mixing matrix, or:  a=rand(N);
x=a*s;                               % mix input signals (permuted)
mixes=a*sources;                     % make mixed sources (not permuted)

%**** if you are loading already-mixed sources:

% mixes=readsounds(['mix2';'mix1']);  % see "help readsounds"

%**** sphere the data
mx=mean(mixes'); c=cov(mixes');
x=x-mx'*ones(1,P);                   % subtract means from mixes
wz=2*inv(sqrtm(c));                  % get decorrelating matrix
x=wz*x;                              % decorrelate mixes so cov(x')=4*eye(N);

%**** 
%w=[1 1; 1 2];                       % init. unmixing matrix, or w=rand(M,N);
w=eye(N);                            % init. unmixing matrix, or w=rand(M,N);
M=size(w,2);                            % M=N usually
sweep=0; oldw=w; olddelta=ones(1,N*N);
Id=eye(M);

%************* this learns: "help sep" explains all 

L=0.01; B=30; sep    % should converge on 1 pass for 2->2 net
L=0.001; B=30; sep   % but annealing will improve soln even more 
L=0.0001; B=30; sep  % and so on

%for multiple sweeps:
L=0.005; B=30; for I=1:10000, sep; end
%***************************************

mixes=a*sources;       % make mixed sources
sound(mixes(1,:))      % play the first one (if it is audio)
plot(mixes(1,:))       % plot the first one (if it is another signal)
uu=w*wz*mixes;            % make unmixed sources
sound(uu(1,:))         % play the first one (if it is audio)
plot(uu(2,:))          % plot the first one (if it is another signal)
