struct Graph
    n :: Int # |V|
    m :: Int # |E|
    u :: Array{Int, 1}
    v :: Array{Int, 1} # uv is an edge
#    w :: Array{Float64, 1} # weight of each edge
    nbr :: Array{Array{Int, 1}, 1}
end

struct Graph_direct
    n :: Int # |V|
    m :: Int # |E|
    u :: Array{Int, 1}
    v :: Array{Int, 1} # uv is an edge
#    w :: Array{Float64, 1} # weight of each edge
    nbr_in  :: Array{Array{Int, 1}, 1}
    nbr_out :: Array{Array{Int, 1}, 1}
end

include("core.jl")

function get_graph_union(G,S)
    nG=G.n;
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    #E = Set{Tuple{Int32, Int32, Float32}}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1

    u = Int[]
    v = Int[]

    tot = 0
    for i = 1 : G.m
        x   = G.u[i]
        y   = G.v[i]
        if x in S
            x=nG+1
        end
        if y in S
            y=nG+1
        end
        if x!=y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y
            push!(u, u1)
            push!(v, v1)
            tot += 1
        end
    end
    nbr=[ [ ] for i in 1:n ]
    for i=1:tot
        u1=u[i];
        v1=v[i];
        push!(nbr[u1],v1);
        push!(nbr[v1],u1);
    end
    return Graph(n, tot, u, v,nbr)
end


function get_graph(ffname)
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    #E = Set{Tuple{Int32, Int32, Float32}}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1

    #fname = string("output/",ffname)
    fname = string("data/",ffname)
    fin = open(fname, "r")


    str = readline(fin)
    str = split(str)
    #n   = parse(Int, str[1])
    m   = parse(Int, str[3])
    u = Int[]
    v = Int[]

    tot = 0
    for i = 1 : m
        str = readline(fin)
        str = split(str)
        x   = parse(Int, str[1])
        y   = parse(Int, str[2])
        if x!=y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y
            push!(u, u1)
            push!(v, v1)
            tot += 1
        end
    end
    nbr=[ [ ] for i in 1:n ]
    for i=1:tot
        u1=u[i];
        v1=v[i];
        push!(nbr[u1],v1);
        push!(nbr[v1],u1);
    end

    close(fin)
    return Graph(n, tot, u, v,nbr)
end



function get_graph_tmp(ffname)
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    #E = Set{Tuple{Int32, Int32, Float32}}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1

    fname = string("output/",ffname)
    fin = open(fname, "r")


    str = readline(fin)
    str = split(str)
    #n   = parse(Int, str[1])
    m   = parse(Int, str[3])

    u = Int[]
    v = Int[]

    tot = 0
    for i = 1 : m
        str = readline(fin)
        str = split(str)
        x   = parse(Int, str[1])
        y   = parse(Int, str[2])
        if x!=y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y
            push!(u, u1)
            push!(v, v1)
            tot += 1
        end
    end
    nbr=[];

    t = m÷10;
    uall=[];
    vall=[];
    for k=1:10
        index1=t*(k-1)+1;
        index2=t*k;
        if k==10
            push!(uall,u[index1:end])
            push!(vall,v[index1:end])
            continue
        end
        push!(uall,u[index1:index2])
        push!(vall,v[index1:index2])
    end

    Gall=[];
    for kk=1:10
        push!(Gall,Graph(n, size(uall[kk])[1], uall[kk], vall[kk],nbr))
    end

    close(fin)
    return Gall
end







function get_graph2(ffname)
    n = 0
    # Label = Dict{Int32, Int32}()
    # Origin = Dict{Int32, Int32}()
    # #E = Set{Tuple{Int32, Int32, Float32}}()

    # getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1

    fname = string("data/",ffname)
    fin = open(fname, "r")


    str = readline(fin)
    str = split(str)
    n   = parse(Int, str[1])
    m   = parse(Int, str[3])
    u = Int[]
    v = Int[]

    tot = 0
    for i = 1 : m
        str = readline(fin)
        str = split(str)
        x   = parse(Int, str[1])
        y   = parse(Int, str[2])
        if x!=y
            # u1 = getID(x)
            # v1 = getID(y)
            # Origin[u1] = x
            # Origin[v1] = y
            push!(u, x)
            push!(v, y)
            tot += 1
        end
    end
    nbr=[ [ ] for i in 1:n ]
    for i=1:tot
        u1=u[i];
        v1=v[i];
        push!(nbr[u1],v1);
        push!(nbr[v1],u1);
    end

    close(fin)
    return Graph(n, tot, u, v,nbr)
end

function get_graph_direct(ffname)
    n = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    #E = Set{Tuple{Int32, Int32, Float32}}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = n += 1

    fname = string("data/",ffname)
    fin = open(fname, "r")


    str = readline(fin)
    str = split(str)
    #n   = parse(Int, str[1])
    m   = parse(Int, str[3])
    u = Int[]
    v = Int[]

    tot = 0
    for i = 1 : m
        str = readline(fin)
        str = split(str)
        x   = parse(Int, str[1])
        y   = parse(Int, str[2])
        if x!=y
            u1 = getID(x)
            v1 = getID(y)
            Origin[u1] = x
            Origin[v1] = y
            push!(u, u1)
            push!(v, v1)
            tot += 1
        end
    end
    nbr_in=[ [ ] for i in 1:n ]
    nbr_out=[ [ ] for i in 1:n ]
    for i=1:tot
        u1=u[i];
        v1=v[i];
        push!(nbr_out[u1],v1);
        push!(nbr_in[v1],u1);
    end

    close(fin)
    return Graph_direct(n, tot, u, v,nbr_in,nbr_out)
end

function findconnect_direct_strong(G)
    a=adjsp_direct(G)
    c=components(a)
    num=zeros(G.n)
    for i=1:G.n
        num[c[i]]+=1;
    end
    lc=argmax(num)

    nc = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    #E = Set{Tuple{Int32, Int32, Float32}}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = nc += 1


        #nc=size(copset)[1];
    mc=0;
    uc = Int[]
    vc = Int[]
    for i=1:G.m
        if c[G.u[i]]==lc
                mc+=1;
                u1 = getID(G.u[i])
                v1 = getID(G.v[i])
                Origin[u1] =  G.u[i]
                Origin[v1] =  G.v[i]
                push!(uc,u1)
                push!(vc,v1)
            end
    end
    nbr_in=[ [ ] for i in 1:n ]
    nbr_out=[ [ ] for i in 1:n ]
    for i=1:mc
        u1=uc[i];
        v1=vc[i];
        push!(nbr_out[u1],v1);
        push!(nbr_in[v1],u1);
    end
        #u=uc;
        #v=vc;
    return Graph(nc,mc,uc,vc,nbr_in,nbr_out);
end


function findconnect(G);
    n=G.n;
    bcj=zeros(G.n);
    idt=0;
    noc=zeros(G.n);
    L=lapsp(G);
    while minimum(bcj)==0
        idt=idt+1;
        #=
        label=0;
        for i=1:G.n
            if bcj[i]==0
                label=i;
                break;
            end
        end
        =#
        label=argmin(bcj);
        b=zeros(n);
        b[1]=label;
        bcj[label]=idt;
        noc[Int(idt)]=1;
        f=1;r=2;
        while f<r
            for i=1:size(G.nbr[Int(b[f])])[1]
                if bcj[Int(G.nbr[Int(b[f])][i])]==0
                    b[r]=G.nbr[Int(b[f])][i];
                    bcj[Int(G.nbr[Int(b[f])][i])]=idt;
                    noc[Int(idt)]+=1;
                    r=r+1;
                end
            end
            f=f+1;
        end
    end
    cop=argmax(noc);
    copset=union([]);


    nc = 0
    Label = Dict{Int32, Int32}()
    Origin = Dict{Int32, Int32}()
    #E = Set{Tuple{Int32, Int32, Float32}}()

    getID(x :: Int) = haskey(Label, x) ? Label[x] : Label[x] = nc += 1


    #nc=size(copset)[1];
    mc=0;
    uc = Int[]
    vc = Int[]
    for i=1:G.m
        if bcj[G.u[i]]==cop
            mc+=1;
            u1 = getID(G.u[i])
            v1 = getID(G.v[i])
            Origin[u1] =  G.u[i]
            Origin[v1] =  G.v[i]
            push!(uc,u1)
            push!(vc,v1)
        end
    end

    nbr=[ [ ] for i in 1:nc ]
    for i=1:mc
        u1=uc[i];
        v1=vc[i];
        push!(nbr[u1],v1);
        push!(nbr[v1],u1);
    end
    #u=uc;
    #v=vc;
    return Graph(nc,mc,uc,vc,nbr);
end
