using ITensors

function get_ising_1d(J::Float64, interactions, sites)

    os = OpSum()
    for k=1:ncol(interactions)
        i = interactions[!,k][1][1]
        j = interactions[!,k][1][2]
        os += 4*J,"Sz",i,"Sz",j
    end
    H = MPO(os,sites)

    return H
end


function get_tfim_1d(L::Int, J::Float64, h::Float64, interactions, sites)

    # Build Hamiltonian
    os = OpSum()
    for k=1:ncol(interactions)
        i = interactions[!,k][1][1]
        j = interactions[!,k][1][2]
        os += 4*J,"Sz",i,"Sz",j
    end
    for k=1:L
        os += 2*h,"Sx",k
    end
    H = MPO(os,sites)
    
    return H
end

function get_afmheis_1d(J::Float64, interactions, sites)

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
    
    return H
end

function get_hamiltonian_1d(which_Ham::String, L::Int, interactions, sites)
    
    if isnothing(sites)
        sites = siteinds("S=1/2",L)
    end
    psi0 = randomMPS(sites, 40)

    if which_Ham=="Ising"
        H = get_ising_1d(-1., interactions, sites)
        save_path = "../out/ising_fm_1d_L$(L)"
    elseif which_Ham=="TFIM"
        H = get_tfim_1d(L, -1., -1., interactions, sites)
        save_path = "../out/ising_fm_1d_L$(L)_h1"
    elseif which_Ham=="Heisenberg"
        H = get_afmheis_1d(+1.,interactions, sites)
        save_path = "../out/heis_afm_1d_L$(L)"
    else
        println(
            "Hamiltonian must be Ising, TFIM, or Heisenberg."
        )
        return
    end

    return H, psi0, save_path
end

#---------------------- 2D Scripts -----------------------------------------

function get_tfim_2d(J::Float64, h::Float64, lattice, sites)

    # Build Hamiltonian
    ampo = OpSum()
    for b in lattice
      ampo .+= 4*J, "Sz", b.s1, "Sz", b.s2
    end
    for site in 1:N
      ampo .+= 2*J, "Sx", site
    end
    H = MPO(ampo,sites)

    return H
end

function get_afmheis_2d(J::Float64, lattice, sites)

    # Build Hamiltonian
    ampo = OpSum()
    for b in lattice
      ampo .+= 2, "S+", b.s1, "S-", b.s2
      ampo .+= 2, "S-", b.s1, "S+", b.s2
      ampo .+= 4,  "Sz", b.s1, "Sz", b.s2
    end
    H = MPO(ampo,sites)
    
    return H
end

function get_hamiltonian_2d(which_Ham::String, Lx::Int, Ly::Int, lattice, sites)
    
    N = Lx*Ly
    if isnothing(sites)
        sites = siteinds("S=1/2", N)
    end
    psi0 = randomMPS(sites,20)

    if which_Ham=="TFIM"
        H = get_tfim_2d(-1., -1., lattice, sites)
        save_path = "../out/ising_fm_2d_L$(Lx)_h1"
    elseif which_Ham=="Heisenberg"
        H = get_afmheis_2d(+1.,lattice, sites)
        save_path = "../out/heis_afm_2d_L$(Lx)"
    else
        println(
            "Hamiltonian must be TFIM, or Heisenberg."
        )
        return
    end

    return H, psi0, save_path
end