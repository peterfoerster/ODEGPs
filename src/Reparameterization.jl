function argext(x::AbstractVector{<:Real}; reltol::Real=1e-3)
    x_max = maximum(abs.(x))

    i_ext = Int[]
    p     = false
    s     = signbit(x[2] - x[1])

    for j in range(2, length(x)-1)
        sp = signbit(x[j+1] - x[j])

        if ~p && ((s && ~sp) || (~s && sp))
            # append candidate extremum
            append!(i_ext, j)
            p = true
        elseif ((s && ~sp) || (~s && sp))
            dx = abs(x[j] - x[i_ext[end]]) / x_max

            # decide about candidate
            if dx >= reltol
                append!(i_ext, j)
            else
                deleteat!(i_ext, length(i_ext))
                p = false
            end
        end

        s = sp
    end

    return i_ext
end

function reparameterize(t::AbstractVector{<:Real}, x::AbstractVector{<:Real}, f::VectorOrNothing=nothing; nu::Int=2, mu::Real=Inf, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, reltol_ext::Real=1e-3, arclength::Bool=true)
    if length(x) != length(t)
        throw(DomainError([length(x) length(t)], "only supports length(x) = length(t)."))
    end

    if ~isnothing(f) && (length(f) != length(t))
        throw(DomainError([length(f) length(t)], "only supports length(f) = length(t)."))
    end

    if ~(nu in [0, 1, 2])
        throw(DomainError(nu, "only supports nu in [0, 1, 2]."))
    end

    if ~(mu in [Inf, 1, 2])
        throw(DomainError(mu, "only supports mu in [Inf, 1, 2]."))
    end

    if ~(integrator in ["ImplicitEuler", "Rosenbrock23", "Rodas4"])
        throw(DomainError(integrator, """only supports integrator in ["ImplicitEuler", "Rosenbrock23", "Rodas4"]."""))
    end

    if abstol <= 0
        throw(DomainError(abstol, "only supports abstol > 0."))
    end

    if reltol <= 0
        throw(DomainError(reltol, "only supports reltol > 0."))
    end

    if reltol_ext <= 0
        throw(DomainError(reltol_ext, "only supports reltol_ext > 0."))
    end

    if isnothing(f)
        if nu == 0
            xh = DI.LinearInterpolation(x, t)
        elseif nu == 1
            # xh = DI.QuadraticSpline(x, t)

            # avoid unphysical oscillations
            xh = DI.PCHIPInterpolation(x, t)
        elseif nu == 2
            xh = DI.CubicSpline(x, t)
        end
    else
        if nu == 0
            throw(DomainError(nu, "only supports nu = 0 if isnothing(f) = true."))
        elseif nu == 1
            xh = DI.CubicHermiteSpline(f, x, t)
        elseif nu == 2
            xh = C2HermiteSpline(t, x, f)
        end
    end

    fh = t -> DI.derivative(xh, t)

    if arclength
        ft = (u, p, t) -> sqrt(1 + fh(t)^2)
    else
        ft = (u, p, t) -> abs(fh(t))
    end

    J_ft = (u, p, t) -> 0

    F = ODE.ODEFunction{false}(ft, jac=J_ft)
    T = (t[1], t[end])
    P = ODE.ODEProblem(F, 0, T)

    if integrator == "ImplicitEuler"
        S = ODE.solve(P, ODE.ImplicitEuler(), dense=false, abstol=abstol, reltol=reltol, maxiters=Int(1e6))
    elseif integrator == "Rosenbrock23"
        S = ODE.solve(P, ODE.Rosenbrock23(), dense=false, abstol=abstol, reltol=reltol, maxiters=Int(1e6))
    elseif integrator == "Rodas4"
        S = ODE.solve(P, ODE.Rodas4(), dense=false, abstol=abstol, reltol=reltol, maxiters=Int(1e6))
    end

    if SciMLBase.successful_retcode(S.retcode)
        tt  = S.t
        tau = S[:]
    else
        throw(ErrorException("ODE solution failed."))
    end

    # ensure monotonicity, min avoids issues due to rounding errors
    if nu == 0
        tauh  = DI.LinearInterpolation(tau, tt)
        taui  = DI.LinearInterpolation(tt, tau)
        tauip = s -> 1 / ft(nothing, nothing, min.(taui(s), t[end]))
    elseif nu == 1
        tauh  = DI.PCHIPInterpolation(tau, tt)
        taui  = DI.PCHIPInterpolation(tt, tau)
        tauip = s -> 1 / ft(nothing, nothing, min.(taui(s), t[end]))
    elseif nu == 2
        tauh  = QuinticMonotonicSpline(tt, tau, ft.(nothing, nothing, tt))
        taui  = QuinticMonotonicSpline(tau, tt, 1 ./ ft.(nothing, nothing, tt))
        tauip = s -> 1 / ft(nothing, nothing, min.(taui(s), t[end]))
    end

    i_ext = argext(x, reltol=reltol_ext)
    t_ext = [t[1]; t[i_ext]; t[end]]
    dt    = diff(t_ext)

    for i in range(1, length(i_ext))
        f_ext = (t, p) -> fh(t)
        F     = BNS.IntervalNonlinearFunction(f_ext)
        P     = BNS.IntervalNonlinearProblem(F, [t_ext[i+1] - dt[i]/2, t_ext[i+1] + dt[i+1]/2])

        # do nothing if signs at boundaries match
        if signbit(fh(t_ext[i+1] - dt[i]/2)) != signbit(fh(t_ext[i+1] + dt[i+1]/2))
            S = BNS.solve(P, BNS.ITP(), maxiters=1000, abstol=1e-16, reltol=1e-16)

            if SciMLBase.successful_retcode(S.retcode)
                t_ext[i+1] = S.u
            else
                throw(ErrorException("Root finding failed."))
            end
        end
    end

    # sigma: [0, 1] -> [0, tau[end]]
    if mu == Inf
        # reparameterize with time
        # sigma  = DI.LinearInterpolation(tauh(t_ext), (t_ext .- t[1]) / (t[end] - t[1]))
        sigma  = DI.LinearInterpolation([0, tau[end]], [0, 1])
        sigmap = s -> DI.derivative(sigma, s)

        # reparameterize with arc length
        # sigma  = s -> tau[end] * s
        # sigmap = s -> tau[end]
    elseif mu == 1
        # reparameterize with time
        sigma = DI.CubicHermiteSpline(zeros(length(t_ext)), tauh(t_ext), (t_ext .- t[1]) / (t[end] - t[1]))

        # reparameterize with arc length
        # sigma  = DI.CubicHermiteSpline(zeros(length(t_ext)), tauh(t_ext), tauh(t_ext) / tau[end])
        sigmap = s -> DI.derivative(sigma, s)
    elseif mu == 2
        # reparameterize with time
        sigma = DI.QuinticHermiteSpline(zeros(length(t_ext)), zeros(length(t_ext)), tauh(t_ext), (t_ext .- t[1]) / (t[end] - t[1]))

        # reparameterize with arc length
        # sigma  = DI.QuinticHermiteSpline(zeros(length(t_ext)), zeros(length(t_ext)), tauh(t_ext), tauh(t_ext) / tau[end])
        sigmap = s -> DI.derivative(sigma, s)
    end

    # min avoids issues due to rounding errors
    tauit  = s -> taui(min.(sigma(s), tau[end]))
    xt     = s -> xh(min.(tauit(s), t[end]))
    tauitp = s -> sigmap(s) * tauip(min.(sigma(s), tau[end]))
    s_ext  = (t_ext .- t[1]) / (t[end] - t[1])

    return (tauit=tauit, xt=xt, tauitp=tauitp, s_ext=s_ext)
end

function hausdorff_distance(Xh::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})
    if size(X, 1) != size(Xh, 1)
        throw(DomainError([size(X, 1) size(Xh, 1)], "only supports size(X, 1) = size(Xh, 1)."))
    end

    d_Hh = hausdorff(Xh, X)
    d_H  = hausdorff(X, Xh)
    d_H  = max(d_Hh, d_H)

    return d_H
end

function hausdorff(Xh::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})
    # d_H = max_xh min_x d(xh, x)
    d_H = 0

    for i in range(1, size(Xh, 2))
        # min_x
        dh = Inf

        for j in range(1, size(X, 2))
            d = LA.norm(Xh[:,i] - X[:,j], 2)

            if d < dh
                dh = d
            end
        end

        # max_xh
        if dh > d_H
            d_H = dh
        end
    end

    return d_H
end
