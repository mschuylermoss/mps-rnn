using DataFrames

function spinchain_open(L)
    
    interactions = DataFrame(["$spin"=>[] for spin in 1:L-1])
    
    for spin in 1:L-1
        neighbor = spin+1
        push!(interactions[!,"$spin"],[spin,neighbor])
    end

    return interactions
end

function spinchain_peri(L)
    
    interactions = DataFrame(["$spin"=>[] for spin in 1:L])
    
    for spin in 1:L-1
        neighbor = spin+1
        push!(interactions[!,"$spin"],[spin,neighbor])
    end
    
    push!(interactions[!,"$L"],[10,1])
    
    return interactions
end

function square_lattice_open(Lx::Int, Ly::Int, bc::String)
    interactions = Vector{Int}[];
    interactions = Vector{Int}[];

    for n in (1: Lx-1)
        for n_ in (1: Ly)
            # horizontal square lattice interactions (excluding boundary terms)
            push!(interactions,[coord_to_site_bravais(Lx, n, n_), 
                                coord_to_site_bravais(Lx, n + 1, n_)])
        end
    end
    for n in (1: Lx)
        for n_ in (1: Ly-1)
            # vertical square lattice interactions (excluding boundary terms)
            push!(interactions,[coord_to_site_bravais(Lx, n, n_), 
                                coord_to_site_bravais(Lx, n, n_ + 1)])
        end
    end
    return interactions
end

function triangular_lattice_open(Lx::Int, Ly::Int, bc::String)
    interactions = Vector{Int}[];

    for n in (1: Lx-1)
        for n_ in (1: Ly)
            # horizontal square lattice interactions (excluding boundary terms)
            push!(interactions,[coord_to_site_bravais(Lx, n, n_), 
                                coord_to_site_bravais(Lx, n + 1, n_)])
        end
    end
    for n in (1: Lx)
        for n_ in (1: Ly-1)
            # vertical square lattice interactions (excluding boundary terms)
            push!(interactions,[coord_to_site_bravais(Lx, n, n_), 
                                coord_to_site_bravais(Lx, n, n_ + 1)])
        end
    end
    for n in (2: Lx)
        for n_ in (2: Ly)
            # diagonal interactions
            push!(interactions,[coord_to_site_bravais(Lx, n, n_), 
            coord_to_site_bravais(Lx, n - 1, n_ - 1)])
        end
    end

    return interactions
end