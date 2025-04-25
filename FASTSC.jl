include("graph.jl")
include("core.jl")
include("bb.jl")
include("schur.jl")

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
B=getB(G);

kk=50;
t1=time();


eps=0.3;
t=Int(round(log(n)/eps^2)+1);


Z=zeros(t,n);
f=approxchol_lap(A);
for i=1:t
    Q=randn(1,m);
    QB=Q*B;
    Z[i,:]=f(QB[1,:]);
end
new_lab=Z'; 



K=200;
index_this=bb(new_lab,K);

ll=length(index_this);



S=index_this;
p=100;
len=200;
Anew,W,We,rat,lavg=init(G,S,p,len);
#println(Anew)
ans=zeros(kk,2);

for jjj=1:kk
    e=zeros(ll);
    f2=approxchol_lap(Anew);
    dis=0;
    pairx=0;pairy=0;
    for i=1:ll
        for j=i+1:ll
            e.=0;
            e[i]=1;e[j]=-1;
            tmp_dis=e'*f2(e);
            if tmp_dis>dis
                dis=tmp_dis;
                pairx=i;
                pairy=j;
            end
        end
    end
    pairx1=index_this[pairx];
    pairy1=index_this[pairy];
    if jjj%10==0
        t3=time();
        println(t3-t1);
    end
    ans[jjj,1]=pairx1;ans[jjj,2]=pairy1;
    Anew[pairx,pairy]=Anew[pairx,pairy]+1;Anew[pairy,pairx]=Anew[pairy,pairx]+1;

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
