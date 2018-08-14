function [obj,Q,P] = l1g_JSRDR(X,W,para)
%min sum_i sum_j |x_i-_x_j*Q*P|_2,p*W(i,j)+alfa*|P|_F^2;
%S.T.                 Q'*Q=I


% function [model,Q,W] = SAIR(X,Y,para)

% % Reference:
% % Multimedia Event Detection Using A Classifier-specific Intermediate Representation. 
% % Zhigang Ma, Yi Yang, Nicu Sebe, Kai Zheng and Alex Haup tmann. 
% % IEEE Transactions on Multimedia. 
% %
% %
% %
% % input: X, num*dim; data matrix obtained by KPCA
% %                    this matrix contains both MED data and SIN data
% %
% %        Y, num*class; label matrix of SIN and MED video 
% %                        
% %
% %        para.r, dimension of the intermedia representation 
% %                Note: r should be smaller than c+1, where c is the number of the concepts 
% %                
% %                 
% %        para.alpha, regularization parameter
% %        para.p, p-norm
%
%
% % output: Q, dim*r 
% %         W, r*class;
% %         model, dim*class, the integration of dimension reduction and
% %         classification.
% 
% % *IMPORTANT*: 
% % KPCA with chi^2 kernel MUST be performed to process the data for event detection
% % Please see the paper for details. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha=para.alpha;
r=para.r;
p=para.p;
[num,dim] = size(X);
Ir = eye(r);
Id = eye(dim);
Q = rand(dim,r);
P = rand(r,dim);
iter = 1;
obji = 1;
eigval_all=zeros(dim,20);
G=zeros(size(W));
d1=ones(dim,1);
while 1
    for i=1:size(X,1)
        for j=1:size(X,1)
            if W(i,j)~=0
               Dt = X(i,:)-X(j,:)*Q*P;
              % G(i,j)=(p/2)./(sqrt(sum(Dt.*Dt,2)+eps).^(2-p));%论文（9）式
              G(i,j)=1/(sqrt(sum(Dt.*Dt,2)+eps).^(2-p));%论文（9）式
            end
        end
    end
    F=W.*G;F=(F+F')/2;%（10式子）
    D=diag(sum(F,1));%（11）式子
    
    Zi=real(sqrt(sum(P.*P,2)+eps));
    d1=1./(Zi);
    Z=spdiags(d1,0,r,r);
  %  obj(iter)=trace(X'*D*X-2*P'*Q'*X'*(F*X-D*X*Q*P))+alpha*trace(Q'*Z*Q);%
  % obj(iter)=trace(X'*D*X-2*P'*Q'*X'*(F*X-D*X*Q*P)+alpha*P'*Z*P);%（12）式    
   obj(iter)=trace(X'*D*X-2*P'*Q'*X'*(F*X-D*X*Q*P)+alpha*P'*P);%（12）式    
   
  
    
    
    U = X'*D*X+alpha*Id;
    V = X'*F*X*X'*F*X;
    [eigvec,eigval] = eig(inv(U)*V);
    eigval_all(:,iter)= diag(eigval);
    [eigval,idx] = sort(real(diag(eigval)),'descend');%descend ascend
    Q = eigvec(:,idx(1:r));%求出（16）式子最大特征值对应的特征向量
    

    
%     Q = orth(Q);QI{iter}=Q;
  %  P = inv(Q'*X'*D*X*Q+alpha*Z)*Q'*X'*F*X;%（14）式求出P
  P = inv(Q'*X'*D*X*Q)*Q'*X'*F*X;%（14）式求出P
    %------------------------------------------------------convergence
%     for i=1:size(X,1)
%         for j=1:size(X,1)
%             if W(i,j)~=0
%                Dt = X(i,:)-X(j,:)*Q*P;
%                G(i,j)=(p/2)./(sqrt(sum(Dt.*Dt,2)+eps).^(2-p));
%             end
%         end
%     end
%     F=W.*G;F=(F+F')/2;
%     D=diag(sum(F,1));
%     obj(iter)=trace(X'*D*X-2*P'*Q'*X'*(F*X-D*X*Q*P)+alpha*P'*P);
%     obj = sum(sqrt(sum(Dt.*Dt,2)+eps).^(2-p))+alpha*(norm(W,'fro')).^2;
   
    cver = abs(real(obj(iter))-real(obji))/real(obji);
    obji = obj(iter); 
    iter = iter+1;
    if ((cver < 10^-3 && iter > 1) || iter == 30), break, end
end
Q = real(Q);
[Q,~]=qr(Q,0);
P = real(P);
% model = Q*W;