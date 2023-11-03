using ITensors

function get_ising_1d(L::Int, J::Float64, interactions)
    sites = siteinds("S=1/2",L)
    psi0 = randomMPS(sites, 40)

    os = OpSum()
    for k=1:ncol(interactions)
        i = interactions[!,k][1][1]
        j = interactions[!,k][1][2]
        os += 4*J,"Sz",i,"Sz",j
    end
    H = MPO(os,sites)

    return H, psi0
end


function get_tfim_1d(L::Int, J::Float64, h::Float64, interactions)
    sites = siteinds("S=1/2",L)
    psi0 = randomMPS(sites, 40)

    # Build Hamiltonian
    os = OpSum()
    for k=1:ncol(interactions)
        i = interactions[!,k][1][1]
        j = interactions[!,k][1][2]
        os += 4*J,"Sz",i,"Sz",j
    end
    for k=1:L
        os -= 2*h,"Sx",k
    end
    H = MPO(os,sites)
    
    return H, psi0
end

function get_afmheis_1d(L::Int, J::Float64, interactions)
    sites = siteinds("S=1/2",L)
    psi0 = randomMPS(sites, 40)

    # Build Hamiltonian
    os = OpSum()
    for k=1:ncol(interactions)
        i = interactions[!,k][1][1]
        j = interactions[!,k][1][2]
    os += 4*J,"Sz",j,"Sz",k
    os += 4*J/2,"S+",j,"S-",k
    os += 4*J/2,"S-",j,"S+",k
    end
    H = MPO(os,sites)
    
    return H, psi0
end