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


function run_dmrg_2d(which_Ham::String, which_Latt::String, snake::Bool, bc::String, Lx::Int, Ly::Int, maxbd::Int, save::Bool)

    if which_Latt == "Square"
        lattice = square_lattice(Lx,Ly; yperiodic=false, snake=snake)
    elseif which_Latt == "Triangular"
        lattice = triangular_lattice(Lx,Ly; yperiodic=false, snake=snake)
    end
    
    H, psi0, save_path = get_hamiltonian_2d(which_Ham, which_Latt, Lx, Ly, lattice, nothing)

    sweeps = Sweeps(20)
    maxdim!(sweeps,20,60,100,100,100,maxbd)
    cutoff!(sweeps,1E-12)
    setnoise!(sweeps, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12)
  
    energy,psi = dmrg(H,psi0,sweeps)
    
    if save   
        save_path = save_path
        println(save_path)
        bond_dim_path = "/chi$(maxbd)"
        if snake
            println("Saving Snaked MPS")
            filename = "/init_snaked.hdf5"
        else
            filename = "/init.hdf5"
        end
        if isdir(save_path*bond_dim_path)
            f = h5open(save_path*bond_dim_path*filename,"w") 
            write(f,"psi",psi)
            close(f)
        else
            if isdir(save_path)
                mkdir(save_path*bond_dim_path)
                f = h5open(save_path*bond_dim_path*filename,"w") 
                write(f,"psi",psi)
                close(f)
            else
                mkdir(save_path)
                mkdir(save_path*bond_dim_path)
                f = h5open(save_path*bond_dim_path*filename,"w") 
                write(f,"psi",psi)
                close(f)
            end
        end
    end
    
    return save_path, psi, energy/(Lx*Ly)
end