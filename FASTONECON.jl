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
    acc2=0;
    acc3=0;
    t1=time();
    
    on=G.n;om=G.m;
    Gc=findconnect(G)
    G=Gc;
    n=G.n;m=G.m;
    A=adjsp(G);
    numei=300;
    f = approxchol_lap(A);
    op = Laplacians.SqLinOp(true,1.0,size(A,1),f);
    e = eigs(op, which=:LM, nev=numei);
    eivalue=e[1];
    eivector=e[2];#n*numei
    neweivalue=zeros(numei);
    for i=1:numei
        neweivalue[i]=eivalue[i]^(1/2);
    end
    matreivalue= diagm(neweivalue);#numei*numei
    new_lab=(matreivalue*eivector');#numei*n
    updatedata=zeros(numei);
    for jjj=1:numei
        updatedata[jjj]=eivalue[jjj]^(-1);
    end

    K=500;
    index_this=bb(new_lab',K);
    ll=length(index_this);

    for ii=1:kk
        neweivalue2=zeros(numei);  
        disall2=0;
        xx=0;yy=0;
        for i in index_this
            for j in index_this
                tmp_dis2=sum((new_lab[:,i] .- new_lab[:,j]).^2);
                if tmp_dis2 > disall2
                    disall2=tmp_dis2;
                    xx=i;yy=j;
                end
            end
        end
        ans[ii,1]=xx;ans[ii,2]=yy;

        for jj=1:numei
            updatedata[jj]=(eivector[:,jj][xx]-eivector[:,jj][yy])^2 + updatedata[jj];
            neweivalue2[jj]=(updatedata[jj])^(-1/2);
        end
        matreivalue2= diagm(neweivalue2);#numei*numei
        new_lab=(matreivalue2*eivector');#numei*n
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

