function downsample(t::AbstractVector{<:Real}, x::AbstractVector{<:Real}; p::Int=1, abstol::Real=1e-16, reltol::Real=1e-4)
    if length(x) != length(t)
        throw(DomainError([length(x) length(t)], "only supports length(x) = length(t)."))
    end

    if p <= 0
        throw(DomainError(p, "only supports p > 0."))
    end

    if abstol <= 0
        throw(DomainError(abstol, "only supports abstol > 0."))
    end

    if reltol <= 0
        throw(DomainError(reltol, "only supports reltol > 0."))
    end

    tb = t[1:1]
    xb = x[1:1]
    i  = 1
    di = 1

    while length(tb) < p
        e_a = abs(x[i] - xb[end])

        if e_a < abstol
            i = i + 1
        else
            append!(tb, t[i])
            append!(xb, x[i])
        end
    end

    while i + di <= length(x)
        e_a = abs(x[i+di] - xb[end])

        if e_a < abstol
            i  = i + di
            di = 1

            continue
        end

        xh  = DI.LagrangeInterpolation([xb[end-p+1:end]; x[i+di]], [tb[end-p+1:end]; t[i+di]])
        e_r = 0

        for j in range(i+1, i+di-1)
            e_r = max(e_r, abs(xh(t[j]) - x[j]) / abs(x[j]))
        end

        if e_r < reltol
            if i + di == length(x)
                append!(tb, t[end])
                append!(xb, x[end])

                break
            else
                di = di+1
            end
        else
            append!(tb, t[i+di-1])
            append!(xb, x[i+di-1])

            i  = i + di - 1
            di = 1
        end
    end

    return (tb=tb, xb=xb)
end

function S_tdo(t_0::Real=0, t_f::Real=1e-2; R::Real=1, L::Real=2e-3, C::Real=100e-9, v_s::Function=t -> 0.25, x_0::AbstractVector{<:Real}=zeros(2), i_x::Int=1, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, directory::String="", filename::String="tdo_t_0=$(t_0)_t_f=$(t_f)_R=$(R)_L=$(L)_C=$(C)_integrator=$(integrator)_abstol=$(abstol)_reltol=$(reltol)")
    # old tdo model
    g_D = v_D -> 10.8 * v_D^2 - 8.766 * v_D + 1.8

    # x' = f(x, t)
    function f_tdo!(f, x, t)
        f[:] = -[1/C * (g_D(x[1]) * x[1] - x[2]); 1/L * (x[1] + R * x[2] - v_s(t))]

        return nothing
    end

    f! = (f, u, p, t) -> f_tdo!(f, u, t)

    if isfile(directory*filename*"_i_x=$(i_x).csv")
        # @warn "ODE solution with filename = $(directory*filename) already exists. Loading this solution."

        S = DF.readdlm(directory*filename*"_i_x=$(i_x).csv", ',')
        t = S[:,1]
        x = S[:,2]
        f = S[:,3]
    else
        F = ODE.ODEFunction{true}(f!)
        t = (t_0, t_f)
        P = ODE.ODEProblem(F, x_0, t)

        # dense=true
        # save_everystep=true
        # adaptive=true
        # abstol=1e-6
        # reltol=1e-3
        # dt, dtmax, dtmin
        # maxiters=Int(1e5)
        # Tsit5/Vern6-9 (nonstiff), ImplicitEuler, Rosenbrock23 (DAE, small), TRBDF2 (medium), Rodas4/5 (DAE, small, accurate), QNDF/FBDF (DAE, large, no oscillations)

        if integrator == "ImplicitEuler"
            S = ODE.solve(P, ODE.ImplicitEuler(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        elseif integrator == "Rosenbrock23"
            S = ODE.solve(P, ODE.Rosenbrock23(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        elseif integrator == "Rodas4"
            S = ODE.solve(P, ODE.Rodas4(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        end

        if SciMLBase.successful_retcode(S.retcode)
            t = S.t
            x = S[:,:]
        else
            throw(ErrorException("ODE solution failed, but Anna loves you."))
        end

        for j in setdiff(range(1, size(x, 1)), i_x)
            f = [S.k[i][1][j] for i in range(1, length(t))]
            DF.writedlm(directory*filename*"_i_x=$(j).csv", [t x[j,:] f], ',')
        end

        x = x[i_x,:]
        f = [S.k[i][1][i_x] for i in range(1, length(t))]
        DF.writedlm(directory*filename*"_i_x=$(i_x).csv", [t x f], ',')
    end

    return (t=t, x=x, f=f)
end

function S_vdpo(t_0::Real=0, t_f::Real=20; mu::Real=1, x_0::AbstractVector{<:Real}=[2, 0], i_x::Int=1, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, directory::String="", filename::String="vdpo_t_0=$(t_0)_t_f=$(t_f)_mu=$(mu)_x_1=$(x_0[1])_x_2=$(x_0[2])_integrator=$(integrator)_abstol=$(abstol)_reltol=$(reltol)")
    function f_vdpo!(f, x, t)
        f[:] = [x[2]; mu * (1 - x[1]^2) * x[2] - x[1]]
    end

    f! = (f, u, p, t) -> f_vdpo!(f, u, t)

    if isfile(directory*filename*"_i_x=$(i_x).csv")
        S = DF.readdlm(directory*filename*"_i_x=$(i_x).csv", ',')
        t = S[:,1]
        x = S[:,2]
        f = S[:,3]
    else
        F = ODE.ODEFunction{true}(f!)
        t = (t_0, t_f)
        P = ODE.ODEProblem(F, x_0, t)

        if integrator == "ImplicitEuler"
            S = ODE.solve(P, ODE.ImplicitEuler(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        elseif integrator == "Rosenbrock23"
            S = ODE.solve(P, ODE.Rosenbrock23(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        elseif integrator == "Rodas4"
            S = ODE.solve(P, ODE.Rodas4(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        end

        if SciMLBase.successful_retcode(S.retcode)
            t = S.t
            x = S[:,:]
        else
            throw(ErrorException("ODE solution failed."))
        end

        for j in setdiff(range(1, size(x, 1)), i_x)
            f = [S.k[i][1][j] for i in range(1, length(t))]
            DF.writedlm(directory*filename*"_i_x=$(j).csv", [t x[j,:] f], ',')
        end

        x = x[i_x,:]
        f = [S.k[i][1][i_x] for i in range(1, length(t))]
        DF.writedlm(directory*filename*"_i_x=$(i_x).csv", [t x f], ',')
    end

    return (t=t, x=x, f=f)
end

function S_brus(t_0::Real=0, t_f::Real=25; k::AbstractVector{<:Real}=[1, 3, 1, 1], x_0::AbstractVector{<:Real}=[1, 1], i_x::Int=1, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, directory::String="", filename::String="brus_t_0=$(t_0)_t_f=$(t_f)_k_1=$(k[1])_k_2=$(k[2])_k_3=$(k[3])_k_4=$(k[4])_x_1=$(x_0[1])_x_2=$(x_0[2])_integrator=$(integrator)_abstol=$(abstol)_reltol=$(reltol)")
    function f_brus!(f, x, t)
        f[:] = [k[1] - (k[2] + k[4]) * x[1] + k[3] * x[1]^2 * x[2]; k[2] * x[1] - k[3] * x[1]^2 * x[2]]
    end

    f! = (f, u, p, t) -> f_brus!(f, u, t)

    if isfile(directory*filename*"_i_x=$(i_x).csv")
        S = DF.readdlm(directory*filename*"_i_x=$(i_x).csv", ',')
        t = S[:,1]
        x = S[:,2]
        f = S[:,3]
    else
        F = ODE.ODEFunction{true}(f!)
        t = (t_0, t_f)
        P = ODE.ODEProblem(F, x_0, t)

        if integrator == "ImplicitEuler"
            S = ODE.solve(P, ODE.ImplicitEuler(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        elseif integrator == "Rosenbrock23"
            S = ODE.solve(P, ODE.Rosenbrock23(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        elseif integrator == "Rodas4"
            S = ODE.solve(P, ODE.Rodas4(), abstol=abstol, reltol=reltol, maxiters=Int(1e6))
        end

        if SciMLBase.successful_retcode(S.retcode)
            t = S.t
            x = S[:,:]
        else
            throw(ErrorException("ODE solution failed."))
        end

        for j in setdiff(range(1, size(x, 1)), i_x)
            f = [S.k[i][1][j] for i in range(1, length(t))]
            DF.writedlm(directory*filename*"_i_x=$(j).csv", [t x[j,:] f], ',')
        end

        x = x[i_x,:]
        f = [S.k[i][1][i_x] for i in range(1, length(t))]
        DF.writedlm(directory*filename*"_i_x=$(i_x).csv", [t x f], ',')
    end

    return (t=t, x=x, f=f)
end
