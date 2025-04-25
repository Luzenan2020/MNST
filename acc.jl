using LinearAlgebra
using Laplacians
using SparseArrays
using Arpack
using Random
using LightGraphs

function accR_large(A,n)
    dis=0;
    pairx=0;
    pairy=0;

    f=approxchol_lap(A);
    e=zeros(n);
    for i=1:n
         for j=i+1:n
             e.=0;
             e[i]=1;e[j]=-1;
             tmp_dis=e'*f(e)
             if tmp_dis>dis
                 dis=tmp_dis;
                 pairx=i;pairy=j;
             end
         end
    end
    return dis;
end

function accR(A,n)
    dis=0;
    pairx=0;
    pairy=0;
    L=zeros(n,n)
    L.=-A;
    for i=1:n
        L[i,i]=sum(A[i,:])
    end

    J=ones(n,n)./n;
    invL=inv(L+J)-J

    for i=1:n
         for j=i+1:n
             tmp_dis=invL[i,i]+invL[j,j]-2*invL[i,j]
             if tmp_dis>dis
                 dis=tmp_dis;
                 pairx=i;pairy=j;
             end
         end
    end
    return dis,pairx,pairy;
end



function accR_all(A,n)
    dis=0;
    pairx=0;
    pairy=0;
    L=zeros(n,n)
    L.=-A;
    for i=1:n
        L[i,i]=sum(A[i,:])
    end
    ans=Float32[];
    J=ones(n,n)./n;
    invL=inv(L+J)-J

    for i=1:n
         for j=i+1:n
             push!(ans,invL[i,i]+invL[j,j]-2*invL[i,j])
             # if tmp_dis>dis
             #     dis=tmp_dis;
             #     pairx=i;pairy=j;
             # end
         end
    end
    #sort!(ans)
    return ans;
end



function mnst1(ans,A)
    kk=size(ans)[1];
    n=size(A)[1];
    acc1=[];
    acc0=0;
    for ii=1:kk
        xx=Int(ans[ii,1]);
        yy=Int(ans[ii,2]);
        f = approxchol_lap(A);
        e=zeros(n);
        e[xx]=1;
        e[yy]=-1;
        tmp_dis=e'*f(e);
        acc0=log(1+tmp_dis)+acc0;
        push!(acc1,acc0);
        A[xx,yy]=1;A[yy,xx]=1;
    end
    return acc1;
end