function [U,x] = boundarylayerFD1d(L,n,eps,f,leftBC,leftBCtype,rightBC,rightBCtype)
%BOUNDARYLAYERFD1D(L,n,f,leftBC,leftBCtype,rightBC,rightBCtype,eps) solves 1d 
%   -u'' + (1/eps)*(u - 1) = 0 using finite difference method.
%
%   BOUNDARYLAYERFD1D solves the equation -u'' + (1/eps)*(u - 1) = 0 on the 
%   domain (0,L) uniformly subdivided into n intervals. Boundary conditions 
%   at x = 0 and x = L are given by leftBC and rightBC and may be Dirichlet 
%   type (indicated 'D') or Neumann type (indicated 'N').
%
%   BOUNDARYLAYERFD1D returns approximate solution U on domain x.
%
%   Author: Tyler Fara           Date: Sept 24, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   EXAMPLES
%   example 1
%       [U,x] = boundarylayerFD1d(1,10,@(x)sin(pi*x),0,'D',0,'D'); plot(x,U);
%
%   example 2
%       f = @(x)(-2*exp(-x))
%       [U,x] = boundarylayerFD1d(2,100,f,-2,'N',0,'D'); plot(x,U);
%
%   example 3 (remove extra line breaks from f when copy/pasting)
%      f = @(x)(-exp(1+(x.^2 - 2*x).^(-1)).*(x.^2 - 2*x).^(-4).*(2*x-2).^2 
%           - 2*exp(1+(x.^2-2*x).^(-1)).*(x.^2-2*x).^(-3).*(2*x-2).^2 
%           + 2*exp(1+(x.^2-2*x).^(-1)).*(x.^2-2*x).^(-2)); 
%      syms y(x); y(x) = piecewise(x<=0,0,0<x&x<2,f(x),2<=x,0);
%      [U,x] = boundarylayerFD1d(1,100,y,0,'N',1,'D'); plot(x,U);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% SETUP AU = F
x = linspace(0, L, n+1);
A = constructDifferenceOperator(x,eps);
F = constructRHS(x,f,eps);


% BOUNDARY CONDITIONS
% error handling
checkBoundaryConditionFormat(leftBCtype,rightBCtype);
checkWellPosedness(leftBCtype,rightBCtype);

% update A and F with boundary conditions
A = updateDifferenceOperator(x,A,leftBCtype,rightBCtype);
F = updateRHS(x,F,leftBC,leftBCtype,rightBC,rightBCtype);


% SOLVE
temp = A \ F';
U = temp';


end


%% DIFFERENCE OPERATOR FUNCTIONS 

function A = constructDifferenceOperator(x,eps)
%CONSTRUCTDIFFERENCEOPERATOR(x) creates matrix A representing the
%   difference operator.

    B = constructSecondDerivativeDifferenceOperator(x);
    C = sparse((1/eps)*eye(length(x)));

    A = B + C;
end


function A = constructSecondDerivativeDifferenceOperator(x)
%CONSTRUCTSECONDDERIVATIVEDIFFERENCEOPERATOR(X) creates matrix A 
%   representing u'' as a finite difference operator.

    % store # of intervals and interval length
    n = length(x)-1;
    h = x(n+1)/n;

    % construct tridiagonal matrix
    mainDiag = 2*ones(n+1,1);
    supDiag  = -1*ones(n,1);
    subDiag  = -1*ones(n,1);
    A = (1/h^2) * sparse(diag(mainDiag,0)+diag(supDiag,1)+diag(subDiag,-1));
end


%% RHS FUNCTIONS

function F = constructRHS(x,f,eps)
%CONSTRUCTRHS(x,f,eps) directs the construction of the right hand side.

    A = computeFunction(x,f);
    B = sparse((1/eps)*ones(1,length(x)));

    F = A + B;
end


function F = computeFunction(x,f)
%STORESOURCE(x,f) evaluates f at x and stores the result in vector F.

    % parse class of f and evaluate f(x) to obtain F
    if isa(f,"double")
        F = f*ones(1,length(x));
    elseif isa(f,"function_handle")
        F = f(x);
    elseif isa(f,"symfun")
        F = f(x);
        F = double(F);
    else
        error 'Incorrect source type. Expected type is source, symfun, or function_handle.'
    end
end


%% BOUNDARY CONDITION FUNCTIONS

function checkBoundaryConditionFormat(leftBCtype,rightBCtype)
%CHECKBOUNDARYCONDITIONFORMAT checks that the boundary condition types are 
%   formatted as 'D' or 'N'.
    
    if (leftBCtype == 'D' || leftBCtype == 'N') ...
        && (rightBCtype == 'D' || rightBCtype == 'N')
        ...
    else
        error 'Incorrect boundary condition types. Input ''D'' or ''N'' for boundary condition type.'
    end
end


function checkWellPosedness(leftBCtype,rightBCtype)
%CHECKWELLPOSEDNESS checks that the problem is not ill-posed.

    if (leftBCtype == 'N' && rightBCtype == 'N')
        error 'Ill-posed problem. Only one boundary may be type ''N''.'
    end
end


function A = updateDifferenceOperator(x,A,leftBCtype,rightBCtype)    
%UPDATEDIFFERENCEOPERATOR updates difference operator A based on the 
%   boundary conditions.

    % store # of intervals and interval length
    n = length(x)-1;
    h = x(n+1)/n;

    % apply x = 0 boundary condition
    if leftBCtype == 'D'
        A(1,1) = 1;
        A(1,2) = 0;
    elseif leftBCtype == 'N'
        A(1,1) = 2/h^2;
        A(1,2) = -2/h^2;
    end
    
    % apply x = L boundary condition
    if rightBCtype == 'D'
        A(n+1,n)   = 0;
        A(n+1,n+1) = 1;
    elseif rightBCtype == 'N'
        A(n+1,n)   = -2/h^2;
        A(n+1,n+1) = 2/h^2;
    end
end


function F = updateRHS(x,F,leftBC,leftBCtype,rightBC,rightBCtype)
%UPDATERHS updates vector F, which represents the right-hand side, based 
%   on the boundary conditions.

    % store # of intervals and interval length
    n = length(x)-1;
    h = x(n+1)/n;

    % apply x = 0 boundary condition
    if leftBCtype == 'D'
        F(1) = leftBC;
    elseif leftBCtype == 'N'
        F(1) = F(1) - (2/h)*leftBC;
    end
    
    % apply x = L boundary condition
    if rightBCtype == 'D'
        F(n+1) = rightBC;
    elseif rightBCtype == 'N'
        F(n+1) = F(n+1) + (2/h)*rightBC;
    end
end