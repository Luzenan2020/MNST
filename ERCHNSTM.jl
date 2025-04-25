include("graph.jl")
include("core.jl")
include("acc.jl")
include("bb.jl")

using LinearAlgebra
using Laplacians
using SparseArrays

fname = open("filename.txt", "r")
str   = readline(fname);
nn    = parse(Int, str);

for ttttt=1:nn

str = readline(fname);
str = split(str);
G   = get_graph(str[1]);    
on=G.n;om=G.m;
Gc=findconnect(G)
G=Gc;
n=G.n;m=G.m;
L=lapsp(G);
A=adjsp(G);
# AA2=adjsp(G);
B=getB(G);
#println(B);
#println(size(B));
t1=time();
kk=50;
ans=zeros(kk,2);

for ii=1:kk 

    eps=0.3;
    t=Int(round(log(n)/eps^2)+1);


    Z=zeros(t,n);
    f=approxchol_lap(A);
    for i=1:t
        Q=randn(1,m);
        QB=Q*B;
        Z[i,:]=f(QB[i,:]);
    end
    new_lab=Z'; 

    disall2=0;
    xx=0;yy=0;


    K=200;
    index_this=bb(new_lab,K);

    ll=length(index_this);


    for i in index_this
        for j in index_this
            tmp_dis2=sum((new_lab[i,:] .- new_lab[j,:]).^2);
            if tmp_dis2 > disall2
                disall2=tmp_dis2;
                xx=i;yy=j;
            end
        end
    end

    ans[ii,1]=xx;ans[ii,2]=yy;


    A[xx,yy]=1;A[yy,xx]=1;
    newadd=zeros(n);
    newadd[xx]=1;newadd[yy]=-1;
    B=vcat(B,newadd');

    m=m+1;

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

close(fname)
