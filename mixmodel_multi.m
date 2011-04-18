function [like,mu,covar,mix]=mixmodel_multi(inputs,M,maxiter,mu,covar,mix)
%
% function [like,mu,covar,mix]=mixmodel(inputs,M,maxiter)
%
% inputs = training data as D dimensional row vectors
% M = number of Gaussians to fit
% maxiter = max number of iterations
% like = vector of log likelihoods at each iteration
%
disp(strcat('starting... (', int2str(M),' clusters)'))

N=size(inputs,1);
D=size(inputs,2);
like=[];
thresh=1e-16;
converged=0;
iter=0;
ll=-inf;
if nargin < 4
    [mu,covar,mix]=randInit(inputs,M);
end
while (iter<maxiter) & ~converged
  prev=ll;
  ll=0;
  for i=1:M
    for n=1:N
        l(n,i) = log(mix(i)) + inputs(n,:)*mu(i,:)';
    end
  end
  for n=1:N
      z = max(l(n,:));
      tau(n,:) = exp(l(n,:)-z)/sum(exp(l(n,:)-z))+1e-5;
  end
  for n=1:N
    k=sum(tau(n,:));
    tau(n,:)=tau(n,:)/k;
    ll=ll+log(k);
  end
  if (nargin >= 4 || ll-prev < thresh)
    converged=1;
  end
  like=[like ll];
  for i=1:M
    sumtau=sum(tau(:,i));
    mu(i,:)=0;
    for n=1:N
      mu(i,:) = mu(i,:) + tau(n,i)*inputs(n,:) / sumtau;
    end
    
    p = sum(mu(i,:));
    
    mu(i,:) = mu(i,:) + (1-p)/N;
    covar(D*(i-1)+1:D*(i-1)+D,:)=0;
    for n=1:N
      covar(D*(i-1)+1:D*(i-1)+D,:) = covar(D*(i-1)+1:D*(i-1)+D,:) + tau(n,i)*(inputs(n,:)-mu(i,:))'*(inputs(n,:)-mu(i,:));
    end
    covar(D*(i-1)+1:D*(i-1)+D,:) = covar(D*(i-1)+1:D*(i-1)+D,:)/sumtau + (1e-5)*eye(D);
    mix(i)=sumtau/N;
  end
  iter=iter+1;
  clf;
  plot(inputs(:,1),inputs(:,2),'g.');
  hold on;
  plotClust(mu,covar,1,2); hold off;
  drawnow;
end

    
    
    
    
