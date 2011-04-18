function [like,alphas,mix]=mixmodel_multi(inputs,M,maxiter)
%
% function [like,alphas,mix]=mixmodel(inputs,M,maxiter)
%
% inputs = training data as D dimensional row vectors
% M = number of Gaussians to fit
% maxiter = max number of iterations
% like = vector of log likelihoods at each iteration
%
disp(strcat('starting... (', int2str(M),' clusters)'))

N = size(inputs,1);
D = size(inputs, 2);
iter = 0;
m=mean(inputs')';
for i=1:M
    r=abs(rand(size(m))*5);
    alpha(:,i)=round(m+r);
    alpha(:,i)=alpha(:,i)/sum(alpha(:,i));
    mix(i) = 1/M;
end

%disp(alpha);
thresh = 1e-4;
ll = 0;
prev = -1;

while(abs(ll-prev) > thresh)
    abs(ll-prev)
    prev = ll;
    % for each cluster, compute log likelihood:
    
    for i=1:M;
        for n=1:D;
            l(n,i) = log(mix(i)) + inputs(:,n)' * log(alpha(:,i));
        end
    end
    
    % likelihoods now established, we calculate tau's
    for n=1:D;
        z = max(l(n,:));
        tau(n,:) = exp(l(n,:)-z)/sum(exp(l(n,:)-z))+1e-5;
    end;
        
    for n=1:D;
      k=sum(tau(n,:));
      tau(n,:)=tau(n,:)/k;
      ll=ll+log(k);
    end
    
    disp(tau);
    pause(0.1);
        
    for i=1:M    
        sumtau = sum(tau(:,i));

        for n=1:D
          alpha(n,i) = sum(inputs(n,:)*tau(n,i))/sum(sum(inputs)*tau(:,i));
        end

        mix(i) = sumtau / N;
    end
    disp(mix);
end