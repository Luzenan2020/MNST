include("graph.jl")
include("core.jl")
include("acc.jl")
include("bb.jl")

using LinearAlgebra
using Laplacians
using SparseArrays
using Arpack
using Random
using LightGraphs


fname = open("filename.txt", "r")
str   = readline(fname);
nn    = parse(Int, str);


for ttttt=1:nn
    str = readline(fname);
    str = split(str);
    G   = get_graph(str[1]);
    kk=50;
    #acc1=1;
    acc2=0;
    acc3=0;
    t1=time();
    
    on=G.n;om=G.m;
    Gc=findconnect(G)
    G=Gc;
    n=G.n;m=G.m;
    println(n,"  ",m)
    #L=lapsp(G);
    A=adjsp(G);
    #B=getB(G);
    #J=ones(n,n)./n;
    #invL=inv(L+J)-J
    ans=zeros(kk,2);
    for ii=1:kk
        numei=300;
        f = approxchol_lap(A);
        op = Laplacians.SqLinOp(true,1.0,size(A,1),f);
        e = eigs(op, which=:LM, nev=numei);
        eivalue=e[1];
        eivector=e[2];
        # println(size(eivector));
        # println(eivector[1:4,1]);
        #println(eivalue[300]);
        # eiv,eii=eigs(L,nev=1133);
        # println(size(eii);)
        # println(eii[1:4,1132]);
        # println(1/eiv[1132]);
        #println(size(eivalue));
        neweivalue=zeros(numei);
        for i=1:numei
            neweivalue[i]=eivalue[i]^(1/2);
        end
        matreivalue= diagm(neweivalue);
        new_lab=(matreivalue*eivector');

        disall2=0;
        xx=0;yy=0;
        K=300;
        index_this=bb(new_lab',K);
        ll=length(index_this);
        for i in index_this
            for j in index_this
                tmp_dis2=sum((new_lab[:,i] .- new_lab[:,j]).^2);
                if tmp_dis2 > disall2
                    disall2=tmp_dis2;
                    xx=i;yy=j;
                end
            end
        end
        #acc3=log(1+disall2)+acc3;
        #println(acc3);
        ans[ii,1]=xx;ans[ii,2]=yy;
        #println(xx," ",yy);
        A[xx,yy]=1;A[yy,xx]=1;
        if ii%10==0
            t3=time();
            println(t3-t1);
        end
    end
    t2=time();
    println(t2-t1);
    A=adjsp(G);
    acc=mnst1(ans,A);

    filename = "output.txt"
    open(filename, "w") do file
        println(file, t2-t1);
        for i= 1:kk
            println(file, acc[i])
        end
    end
end

