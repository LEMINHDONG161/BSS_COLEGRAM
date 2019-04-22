function [W, H] = nmf_s(V, K, W, MAXITER, fixedInds)

F = size(V,1); T = size(V,2);

rand('seed',0)
if isempty(W)
    W = 1+rand(F, sum(K));
end
H = 1+rand(sum(K), T);

% rand('seed',0)
% if isempty(H)
%     H = 1+rand(T, sum(K));
% end

inds = setdiff(1:sum(K),fixedInds);
ONES = ones(F,T);

for i=1:MAXITER 
    
    % update activations
    H = H .* (W'*( V./(W*H+eps))) ./ (W'*ONES);
%     H(inds,:) = H(inds,:) .* (W(:,inds)'*( V./(W*H+eps))) ./ (W(:,inds)'*ONES);
%     
    % update dictionaries
    W(:,inds) = W(:,inds) .* ((V./(W*H+eps))*H(inds,:)') ./(ONES*H(inds,:)');
end

% normalize W to sum to 1
sumW = sum(W);
W = W*diag(1./sumW);
H = diag(sumW)*H;

