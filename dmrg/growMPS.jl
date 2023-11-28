using ITensors

include("geometries.jl")
include("Hamiltonians.jl")
include("dmrg.jl")

function grow(smaller_psi,larger_psi)
    
    psi_s = deepcopy(smaller_psi)
    psi_l = deepcopy(larger_psi)
    N_s = length(psi_s)
    N_l = length(psi_l)
    
    # Step 1: Get MPS in center canonical form with middle lambda
    half_s = Int(N_s/2)
    half_l = Int(N_l/2)
    psi_s = orthogonalize(psi_s,half_s)
    psi_l = orthogonalize(psi_l,half_l)
    
    combiner_s = combiner(siteind(psi_s,half_s),linkind(psi_s,half_s-1); tags="site, left link")
    combined_s = combinedind(combiner_s)
    U_s,lambda_s,V_s = svd(psi_s[half_s]*combiner_s,combined_s,lefttags = "alpha$half_s",righttags = "beta$half_s");
    U_s = dag(combiner_s) * U_s;
    center_alpha_s,center_beta_s = inds(lambda_s);
    inv_lambda_s = ITensor(inv(Matrix(lambda_s,center_alpha_s,center_beta_s)),center_beta_s,center_alpha_s);

    combiner_ = combiner(siteind(psi_l,half_l),linkind(psi_l,half_l-1); tags="site, left link")
    combined_ = combinedind(combiner_)
    U_l,lambda_l,V_l = svd(psi_l[half_l]*combiner_,combined_,lefttags = "alpha$half_l",righttags = "beta$half_l")
    U_l = dag(combiner_) * U_l;

    # Re-tag and Re-name matrices
    link_left = commonind(psi_l[half_l - 1],U_l)
    center_alpha = Index(dim(link_left),"alpha$half_s")
    link_right = commonind(V_l*psi_l[half_l + 1],psi_l[half_l + 2])
    center_beta = Index(dim(link_right),"beta$half_s")
            
    replaceind!(inv_lambda_s,center_beta_s,center_beta)
    replaceind!(inv_lambda_s,center_alpha_s,center_alpha)

    AR2 = replaceind!(psi_l[half_l - 1],link_left,center_alpha)
    AR1 = replaceind!(U_l,link_left,center_alpha)
    BL1 = replaceind!(V_l*psi_l[half_l + 1],link_right,center_beta)
    BL2 = replaceind!(psi_l[half_l + 2],link_right,center_beta)
    
    sitepR = prime(inds(BL1,"Site")[1])
    BR = replaceind(BL1,inds(BL1,"Site")[1],sitepR)
    sitepL = prime(inds(AR1,"Site")[1])
    AL = replaceind(AR1,inds(AR1,"Site")[1],sitepL)
    
    # Stitch together the new MPS
    new_MPS = MPS(N_l + 2)
    N_new = length(new_MPS)
    
    for i in 1:(half_l-2)
        new_MPS[i] = psi_l[i]
    end
    
    rightlink = inds(AR2,"alpha$(half_l-1)")[1]
    newrightlink = Index(dim(rightlink),"Link,l=$(half_l-1)")
    new_MPS[half_l-1] = replaceind(AR2,rightlink,newrightlink)
    
    leftlink = rightlink
    newleftlink = newrightlink
    rightlink = inds(AR1*lambda_l,"beta$(half_l)")[1]
    newrightlink = Index(dim(rightlink),"Link,l=$(half_l)")
    new_MPS[half_l] = replaceinds(AR1*lambda_l,(leftlink,rightlink),(newleftlink,newrightlink))
    
    leftlink = rightlink
    newleftlink = newrightlink
    rightlink = inds(BR*inv_lambda_s,"alpha$(half_l-1)")[1]
    newrightlink = Index(dim(rightlink),"Link,l=$(half_l+1)")
    site = inds(noprime(BR*inv_lambda_s),"Site")[1]
    newsiteind = Index(dim(site),"S=1/2,Site,n=$(half_l+1)")
    new_MPS[half_l+1] = replaceinds(noprime(BR*inv_lambda_s),(leftlink,rightlink,site),(newleftlink,newrightlink,newsiteind))

    leftlink = rightlink
    newleftlink = newrightlink
    rightlink = inds(AL*lambda_l,"beta$(half_l)")[1]
    newrightlink = Index(dim(rightlink),"Link,l=$(half_l+2)")
    site = inds(noprime(AL*lambda_l),"Site")[1]
    newsiteind = Index(dim(site),"S=1/2,Site,n=$(half_l+2)")
    new_MPS[half_l+2] = replaceinds(noprime(AL*lambda_l),(leftlink,rightlink,site),(newleftlink,newrightlink,newsiteind))

    leftlink = rightlink
    newleftlink = newrightlink
    rightlink = inds(BL1,"beta$(half_l-1)")[1]
    newrightlink = Index(dim(rightlink),"Link,l=$(half_l+3)")
    site = inds(BL1,"Site")[1]
    newsiteind = Index(dim(site),"S=1/2,Site,n=$(half_l+3)")
    new_MPS[half_l+3] = replaceinds(BL1,(leftlink,rightlink,site),(newleftlink,newrightlink,newsiteind))

    # need to add a flag for if this is the last site ...
    leftlink = rightlink
    newleftlink = newrightlink
    rightlink = inds(BL2,"Link,l=$(half_l+2)")[1]
    newrightlink = Index(dim(rightlink),"Link,l=$(half_l+4)")
    site = inds(BL2,"Site")[1]
    newsiteind = Index(dim(site),"S=1/2,Site,n=$(half_l+4)")
    new_MPS[half_l+4] = replaceinds(BL2,(leftlink,rightlink,site),(newleftlink,newrightlink,newsiteind))
    
    # println(norm(new_MPS[half_l-1]*new_MPS[half_l]*new_MPS[half_l+1]*new_MPS[half_l+2]*new_MPS[half_l+3]*new_MPS[half_l+4]))
    # println(norm(AR2*AR1*lambda_l*BR*inv_lambda_s*AL*lambda_l*BL1*BL2))
    
    for i in (half_l+3):N_l
        siteind = inds(psi_l[i],"Site")[1]
        newsiteind = Index(dim(siteind),"S=1/2,Site,n=$(i+2)")
        leftlink = rightlink
        newleftlink = newrightlink
        new_MPS[i+2] = replaceinds(psi_l[i],(siteind,leftlink),(newsiteind,newleftlink))
        if i < N_l
            rightlink = inds(psi_l[i],"l=$(i)")[1]
            newrightlink = Index(dim(rightlink),"Link,l=$(i+2)")
            new_MPS[i+2] = replaceind!(new_MPS[i+2],rightlink,newrightlink)
        end
    end
        
    return new_MPS
end

function grow_MPS(which_Ham::String, bc::String, maxbd::Int, L_original,L_final,save::Bool)
    
    n_grows = Int((L_final-L_original)/2)    
    _, psi0, _ = run_dmrg_1d(which_Ham, bc, L_original-2, maxbd, false);
    save_path, psi1,_ = run_dmrg_1d(which_Ham, bc, L_original, maxbd, true); 
    
    # Grow the original MPS
    for n in 1:n_grows
        psi_new = grow(psi0,psi1)
        psi0 = psi1
        psi1 = psi_new
    end

    # Estimate energy of enlarged MPS
    if bc == "open"
        interactions = spinchain_open(L_final)
    elseif bc == "peri"
        interactions = spinchain_peri(L_final)
    end
    H_large, _, save_path_large = get_hamiltonian(which_Ham, L_final, interactions, siteinds(psi1))
    enlarged_energy = inner(psi1',H_large,psi1)
    
    # Save enlarged MPS in appropriate directory
    if save   
        save_path_large = save_path_large
        bond_dim_path = "/chi$(maxbd)"
        if isdir(save_path_large*bond_dim_path)
            f = h5open(save_path_large*bond_dim_path*"/init_from_L$(L_original).hdf5","w") 
            write(f,"psi",psi1)
            close(f)
        else
            if isdir(save_path_large)
                mkdir(save_path_large*bond_dim_path)
                f = h5open(save_path_large*bond_dim_path*"/init_from_L$(L_original).hdf5","w") 
                write(f,"psi",psi1)
                close(f)
            else
                mkdir(save_path_large)
                mkdir(save_path_large*bond_dim_path)
                f = h5open(save_path_large*bond_dim_path*"/init_from_L$(L_original).hdf5","w") 
                write(f,"psi",psi1)
                close(f)
            end
        end
    end

    return psi1, enlarged_energy
end