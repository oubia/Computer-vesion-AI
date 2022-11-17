
prwaitbar off

% four 1-D points
X = [1 2 4 6 ]';
%X = [1 2 4 6 1.5]';

% extended 2-D points and labels
Y = [ ones(size(X)) X];
Z = [ -1 -1 1 1]';
%Z = [ -1 -1 1 1 1]';

% an extended weight vector
a = [ .5 -.3]';


 % a range of values
Np = 200;
XX = linspace(0,7,Np)';
YY = [ones(Np,1) XX];

% Compute outputs
% note that input vectors are transposed
% and that (a' * y)' = y' * a
Xout = Y*a;
XXout = YY*a;

% Decision boundary is g(y) = 0
% wx + b = 0
% x = -b/w
xbound = -a(1)/a(2);

% Plot feature space and outputs
figure(1),clf,hold on
%plot(X,Z,'x')
plot(XX,XXout)
plot(X(Z==1),zeros(size(X(Z==1))),'o')
plot(X(Z==-1),zeros(size(X(Z==-1))),'bs')
plot([xbound xbound],ylim,'k--')
grid on
leg = {'g(x;a)','class 1','class 2','Boundary'};
legend(leg)
xlabel('Input space')
ylabel('Outputs')
title('Simple 1D problem with 4 points')


% The error-count criterion
% we use Y*a-eps<0 instead of Y*a<=0
% to account for zeroes with numerical tolerances
Je = @(a) (sum(Z.*(Y*a-eps)<0))
% the same with separate input args
Je2 = @(a1,a2) (Je([a1;a2]));

% just to try
Je2(a(1),a(2))
Je(a)


% a range of possible weights
wRes=100;
w1=linspace(-1,1,wRes);
w2=linspace(-1,1,wRes);
[ww1,ww2]=meshgrid(w1,w2);

WW = [ww1(:) ww2(:)]; % weight space

JJe = arrayfun(Je2,ww1,ww2); % Error surface

figure(2),clf,surf(ww1,ww2,JJe)
xlabel('w_0')
ylabel('w_1')
zlabel('Je')


%Lets consider all weight vectors of norm 1 only
ang=linspace(0,2*pi,wRes+1)';
ang=ang(1:end-1); % discard the last one

JJ1e = arrayfun(Je2,cos(ang),sin(ang));
hold on
plot3(cos(ang),sin(ang),JJ1e,'k.')

title('Perceptron criterion surface')




figure(3),clf
plot(ang,JJ1e)
xlabel('angle')
ylabel('Je')

% find the optima
optima = find(JJ1e==min(JJ1e));
woptima = [cos(ang(optima)) sin(ang(optima))];

% plot them
figure(3),hold on
plot(ang(optima),JJ1e(optima),'k.')

title('Profile of criteria surfaces (for unitary weights)')


% plot boundaries
figure(1),hold on
for k=1:size(woptima,1)
    plot(XX,YY*woptima(k,:)','b:','LineWidth',k)
    leg = [leg ['g(x;a' num2str(k) ')']];
end

legend(leg)


% Now a different criterion
sqr = @(x) x.*x;
% The perceptron criterion
% now <=0 is ok because the difference is adding zeroes
Jp = @(a) (sum((Z.*(Y*a)<=0).*(-(Z.*(Y*a)) )));
%Jq = @(a) (sum((Z.*(Y*a)<=0).*(sqr(Z.*(Y*a)) )));
% the same with separate input args
Jp2 = @(a1,a2) (Jp([a1;a2]));

JJp = arrayfun(Jp2,ww1,ww2); % Error surface

figure(22),clf,surf(ww1,ww2,JJp)
hold on
nLevels=40;
contour(ww1,ww2,JJp,nLevels)
xlabel('w_0')
ylabel('w_1')
zlabel('Je')

JJ1p = arrayfun(Jp2,cos(ang),sin(ang));
hold on
plot3(cos(ang),sin(ang),JJ1p,'k.')

title('Widrow-Hoff criterion surface')




% find the optima
optimaP = find(JJ1p==min(JJ1p));
woptimaP = [cos(ang(optimaP)) sin(ang(optimaP))];


figure(3),hold on
plot(ang,JJ1p,'r--')
plot(ang(optimaP),JJ1p(optimaP),'ko')
ylabel('Criteria')
legend({'Je','min Je','Jp','min Jp'})


% plot boundaries
figure(1),hold on
for k=1:size(woptimaP,1)
    plot(XX,YY*woptimaP(k,:)','g--','LineWidth',k)
    leg = [leg ['g_p(x;a' num2str(k) ')']];
end

legend(leg)



figure(4),clf,contour(ww1,ww2,JJp,nLevels),hold on
plot(cos(ang),sin(ang),'k.')
xlabel('w_0')
ylabel('w_1')


% Compute gradients
% now it is important again to catch the zeros
gradJp  = @(a) (-sum((Z.*(Y*a+eps)<0).*(Z.*Y)));
%gradJq  = @(a) (-sum((Z.*(Y*a+eps)<0).*((Y*a).*Y)));
gradJp2 = @(a1,a2) (gradJp([a1;a2]));
gradJJ1p = cell2mat(arrayfun(gradJp2,cos(ang),sin(ang),'UniformOutput',false));

% plot gradients
figure(4),hold on
quiver(cos(ang),sin(ang),gradJJ1p(:,1),gradJJ1p(:,2))

title('Widrow-Hoff criterion levels')




