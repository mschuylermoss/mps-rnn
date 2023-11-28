using Base.Filesystem
using ITensors

include("geometries.jl")
include("Hamiltonians.jl")

function run_dmrg_1d(which_Ham::String, bc::String, L::Int, maxbd::Int, save::Bool)
    
    if bc == "open"
        interactions = spinchain_open(L)
    elseif bc == "peri"
        interactions = spinchain_peri(L)
    end
    
    H, psi0, save_path = get_hamiltonian_1d(which_Ham, L, interactions, nothing)

    sweeps = Sweeps(10)
    setmaxdim!(sweeps, 10,20,100,100,maxbd)
    setcutoff!(sweeps, 1E-10)

    energy, psi = dmrg(H, psi0, sweeps, outputlevel=1)
    
    if save   
        save_path = save_path
        bond_dim_path = "/chi$(maxbd)"
        if isdir(save_path*bond_dim_path)
            f = h5open(save_path*bond_dim_path*"/init.hdf5","w") 
            write(f,"psi",psi)
            close(f)
        else
            if isdir(save_path)
                mkdir(save_path*bond_dim_path)
                f = h5open(save_path*bond_dim_path*"/init.hdf5","w") 
                write(f,"psi",psi)
                close(f)
            else
                mkdir(save_path)
                mkdir(save_path*bond_dim_path)
                f = h5open(save_path*bond_dim_path*"/init.hdf5","w") 
                write(f,"psi",psi)
                close(f)
            end
        end
    end
    
    return save_path, psi, energy/L
end


function run_dmrg_2d(which_Ham::String, which_Latt::String, bc::String, Lx::Int, Ly::Int, maxbd::Int, save::Bool)

    if which_Latt == "Square"
        lattice = square_lattice(Lx,Ly; yperiodic=false)
    elseif which_Latt == "Triangular"
        lattice = triangular_lattice(Lx,Ly; yperiodic=false)
    end
    
    H, psi0, save_path = get_hamiltonian_2d(which_Ham, Lx, Ly, lattice, nothing)

    sweeps = Sweeps(20)
    maxdim!(sweeps,20,60,100,100,100,maxbd)
    cutoff!(sweeps,1E-12)
    @show sweeps
  
    energy,psi = dmrg(H,psi0,sweeps)
    
    if save   
        save_path = save_path
        bond_dim_path = "/chi$(maxbd)"
        if isdir(save_path*bond_dim_path)
            f = h5open(save_path*bond_dim_path*"/init.hdf5","w") 
            write(f,"psi",psi)
            close(f)
        else
            if isdir(save_path)
                mkdir(save_path*bond_dim_path)
                f = h5open(save_path*bond_dim_path*"/init.hdf5","w") 
                write(f,"psi",psi)
                close(f)
            else
                mkdir(save_path)
                mkdir(save_path*bond_dim_path)
                f = h5open(save_path*bond_dim_path*"/init.hdf5","w") 
                write(f,"psi",psi)
                close(f)
            end
        end
    end
    
    return save_path, psi, energy/(Lx*Ly)
end